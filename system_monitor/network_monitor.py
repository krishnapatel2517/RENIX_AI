"""
RENIX Network Monitor
=====================

Provides network monitoring for RENIX.

Features:
- Network interfaces
- Upload/download counters
- Network connection statistics
- Interface status
- Data transfer history
- Bandwidth speed estimation
- Health analysis

Dependency:
    pip install psutil
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None


class NetworkMonitor:
    """
    Network monitoring component for RENIX.

    Example:
        monitor = NetworkMonitor()

        stats = monitor.get_stats()

        print(stats)
    """

    def __init__(
        self,
        *,
        history_size: int = 120,
    ) -> None:

        self.history_size = max(
            1,
            int(history_size),
        )

        self.history: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.history_size
        )

        self.last_stats: (
            dict[str, Any] | None
        ) = None

        self.last_update_time: (
            float | None
        ) = None

        self._previous_io: (
            dict[str, Any] | None
        ) = None

        self._previous_io_time: (
            float | None
        ) = None

    # ============================================================
    # MAIN API
    # ============================================================

    def get_stats(
        self,
        *,
        store: bool = True,
    ) -> dict[str, Any]:
        """
        Collect complete network statistics.
        """

        if psutil is None:

            return {
                "available": False,
                "error": (
                    "psutil is not installed."
                ),
            }

        try:

            current_time = time.time()

            io_stats = (
                self._get_io_stats()
            )

            interfaces = (
                self._get_interfaces()
            )

            connections = (
                self._get_connection_stats()
            )

            speed = (
                self._calculate_speed(
                    io_stats,
                    current_time,
                )
            )

            stats = {
                "available": True,
                "timestamp": (
                    datetime.now().isoformat(
                        timespec="seconds"
                    )
                ),
                "io": io_stats,
                "interfaces": interfaces,
                "connections": connections,
                "speed": speed,
                "interface_count": len(
                    interfaces
                ),
                "active_interfaces": (
                    sum(
                        1
                        for interface
                        in interfaces
                        if interface.get(
                            "is_up"
                        )
                    )
                ),
            }

            self._previous_io = (
                io_stats.copy()
            )

            self._previous_io_time = (
                current_time
            )

            self.last_stats = stats

            self.last_update_time = (
                current_time
            )

            if store:

                self.history.append(
                    stats
                )

            return stats

        except Exception as exc:

            return {
                "available": False,
                "error": str(exc),
            }

    # ============================================================
    # NETWORK I/O
    # ============================================================

    def _get_io_stats(
        self,
    ) -> dict[str, Any]:
        """
        Get total network I/O statistics.
        """

        if psutil is None:
            return {}

        try:

            io = (
                psutil.net_io_counters()
            )

            if io is None:
                return {}

            return {
                "bytes_sent": int(
                    io.bytes_sent
                ),
                "bytes_received": int(
                    io.bytes_recv
                ),
                "packets_sent": int(
                    io.packets_sent
                ),
                "packets_received": int(
                    io.packets_recv
                ),
                "errors_in": int(
                    io.errin
                ),
                "errors_out": int(
                    io.errout
                ),
                "dropped_in": int(
                    io.dropin
                ),
                "dropped_out": int(
                    io.dropout
                ),
                "sent_mb": round(
                    io.bytes_sent
                    / 1024
                    / 1024,
                    2,
                ),
                "received_mb": round(
                    io.bytes_recv
                    / 1024
                    / 1024,
                    2,
                ),
                "sent_gb": round(
                    io.bytes_sent
                    / 1024
                    / 1024
                    / 1024,
                    2,
                ),
                "received_gb": round(
                    io.bytes_recv
                    / 1024
                    / 1024
                    / 1024,
                    2,
                ),
            }

        except Exception:
            return {}

    # ============================================================
    # SPEED CALCULATION
    # ============================================================

    def _calculate_speed(
        self,
        current_io: dict[str, Any],
        current_time: float,
    ) -> dict[str, Any]:
        """
        Estimate current upload/download speed.
        """

        if (
            self._previous_io is None
            or self._previous_io_time is None
        ):

            return {
                "download_bytes_per_second": 0.0,
                "upload_bytes_per_second": 0.0,
                "download_kbps": 0.0,
                "upload_kbps": 0.0,
                "download_mbps": 0.0,
                "upload_mbps": 0.0,
            }

        elapsed = (
            current_time
            - self._previous_io_time
        )

        if elapsed <= 0:

            return {
                "download_bytes_per_second": 0.0,
                "upload_bytes_per_second": 0.0,
                "download_kbps": 0.0,
                "upload_kbps": 0.0,
                "download_mbps": 0.0,
                "upload_mbps": 0.0,
            }

        previous_received = (
            self._previous_io.get(
                "bytes_received",
                0,
            )
        )

        previous_sent = (
            self._previous_io.get(
                "bytes_sent",
                0,
            )
        )

        current_received = (
            current_io.get(
                "bytes_received",
                0,
            )
        )

        current_sent = (
            current_io.get(
                "bytes_sent",
                0,
            )
        )

        download_speed = max(
            0,
            (
                current_received
                - previous_received
            )
            / elapsed,
        )

        upload_speed = max(
            0,
            (
                current_sent
                - previous_sent
            )
            / elapsed,
        )

        return {
            "download_bytes_per_second": (
                round(
                    download_speed,
                    2,
                )
            ),
            "upload_bytes_per_second": (
                round(
                    upload_speed,
                    2,
                )
            ),
            "download_kbps": (
                round(
                    download_speed
                    * 8
                    / 1000,
                    2,
                )
            ),
            "upload_kbps": (
                round(
                    upload_speed
                    * 8
                    / 1000,
                    2,
                )
            ),
            "download_mbps": (
                round(
                    download_speed
                    * 8
                    / 1_000_000,
                    2,
                )
            ),
            "upload_mbps": (
                round(
                    upload_speed
                    * 8
                    / 1_000_000,
                    2,
                )
            ),
        }

    # ============================================================
    # NETWORK INTERFACES
    # ============================================================

    def _get_interfaces(
        self,
    ) -> list[dict[str, Any]]:
        """
        Collect all network interface information.
        """

        if psutil is None:
            return []

        interfaces = []

        try:

            addresses = (
                psutil.net_if_addrs()
            )

            interface_stats = (
                psutil.net_if_stats()
            )

            for name, address_list in (
                addresses.items()
            ):

                stats = (
                    interface_stats.get(
                        name
                    )
                )

                formatted_addresses = []

                for address in address_list:

                    formatted_addresses.append(
                        {
                            "family": str(
                                address.family
                            ),
                            "address": (
                                address.address
                            ),
                            "netmask": (
                                address.netmask
                            ),
                            "broadcast": (
                                address.broadcast
                            ),
                        }
                    )

                interfaces.append(
                    {
                        "name": name,
                        "is_up": (
                            stats.isup
                            if stats
                            else False
                        ),
                        "speed_mbps": (
                            stats.speed
                            if stats
                            else None
                        ),
                        "mtu": (
                            stats.mtu
                            if stats
                            else None
                        ),
                        "addresses": (
                            formatted_addresses
                        ),
                    }
                )

        except Exception:
            pass

        return interfaces

    # ============================================================
    # CONNECTIONS
    # ============================================================

    def _get_connection_stats(
        self,
    ) -> dict[str, Any]:
        """
        Get network connection statistics.
        """

        if psutil is None:
            return {}

        try:

            connections = (
                psutil.net_connections(
                    kind="inet"
                )
            )

            status_counts: dict[
                str,
                int,
            ] = {}

            for connection in connections:

                status = (
                    connection.status
                    or "NONE"
                )

                status_counts[status] = (
                    status_counts.get(
                        status,
                        0,
                    )
                    + 1
                )

            established = (
                status_counts.get(
                    "ESTABLISHED",
                    0,
                )
            )

            listening = (
                status_counts.get(
                    "LISTEN",
                    0,
                )
            )

            return {
                "total": len(
                    connections
                ),
                "established": established,
                "listening": listening,
                "statuses": (
                    status_counts
                ),
            }

        except (
            PermissionError,
            OSError,
        ):

            return {
                "total": 0,
                "established": 0,
                "listening": 0,
                "statuses": {},
                "restricted": True,
            }

        except Exception:
            return {}

    # ============================================================
    # QUICK METRICS
    # ============================================================

    def get_download_speed(
        self,
    ) -> float:
        """
        Return current estimated download speed in Mbps.
        """

        stats = self.get_stats(
            store=False
        )

        speed = stats.get(
            "speed",
            {},
        )

        return float(
            speed.get(
                "download_mbps",
                0.0,
            )
        )

    def get_upload_speed(
        self,
    ) -> float:
        """
        Return current estimated upload speed in Mbps.
        """

        stats = self.get_stats(
            store=False
        )

        speed = stats.get(
            "speed",
            {},
        )

        return float(
            speed.get(
                "upload_mbps",
                0.0,
            )
        )

    def get_active_interfaces(
        self,
    ) -> list[dict[str, Any]]:
        """
        Return currently active network interfaces.
        """

        stats = self.get_stats(
            store=False
        )

        interfaces = stats.get(
            "interfaces",
            [],
        )

        return [
            interface
            for interface in interfaces
            if interface.get(
                "is_up"
            )
        ]

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> dict[str, Any]:
        """
        Return simplified network status.
        """

        stats = self.get_stats(
            store=False
        )

        if not stats.get(
            "available"
        ):

            return {
                "available": False,
                "status": "unknown",
            }

        active_interfaces = (
            stats.get(
                "active_interfaces",
                0,
            )
        )

        connections = stats.get(
            "connections",
            {},
        )

        established = (
            connections.get(
                "established",
                0,
            )
            if isinstance(
                connections,
                dict,
            )
            else 0
        )

        if active_interfaces == 0:

            status = "offline"

        elif established == 0:

            status = "limited"

        else:

            status = "connected"

        return {
            "available": True,
            "status": status,
            "active_interfaces": (
                active_interfaces
            ),
            "established_connections": (
                established
            ),
            "healthy": (
                status == "connected"
            ),
        }

    # ============================================================
    # ANALYSIS
    # ============================================================

    def analyze(
        self,
    ) -> dict[str, Any]:
        """
        Analyze network health.
        """

        stats = self.get_stats()

        if not stats.get(
            "available"
        ):

            return {
                "available": False,
                "status": "unknown",
                "issues": [],
                "recommendations": [],
            }

        issues: list[str] = []

        recommendations: list[
            str
        ] = []

        active_interfaces = (
            stats.get(
                "active_interfaces",
                0,
            )
        )

        io_stats = stats.get(
            "io",
            {},
        )

        if active_interfaces == 0:

            issues.append(
                "No active network interfaces."
            )

            recommendations.append(
                "Check Wi-Fi or Ethernet connection."
            )

        if isinstance(
            io_stats,
            dict,
        ):

            errors = (
                int(
                    io_stats.get(
                        "errors_in",
                        0,
                    )
                )
                + int(
                    io_stats.get(
                        "errors_out",
                        0,
                    )
                )
            )

            dropped = (
                int(
                    io_stats.get(
                        "dropped_in",
                        0,
                    )
                )
                + int(
                    io_stats.get(
                        "dropped_out",
                        0,
                    )
                )
            )

            if errors > 0:

                issues.append(
                    f"Network errors detected: "
                    f"{errors}"
                )

            if dropped > 0:

                issues.append(
                    f"Network packets dropped: "
                    f"{dropped}"
                )

        if active_interfaces == 0:

            status = "offline"

        elif issues:

            status = "warning"

        else:

            status = "normal"

        return {
            "available": True,
            "status": status,
            "issues": issues,
            "recommendations": (
                recommendations
            ),
            "download_speed_mbps": (
                stats.get(
                    "speed",
                    {},
                ).get(
                    "download_mbps",
                    0.0,
                )
            ),
            "upload_speed_mbps": (
                stats.get(
                    "speed",
                    {},
                ).get(
                    "upload_mbps",
                    0.0,
                )
            ),
        }

    # ============================================================
    # HISTORY
    # ============================================================

    def get_history(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:

        history = list(
            self.history
        )

        if limit is None:
            return history

        limit = max(
            0,
            int(limit),
        )

        if limit == 0:
            return []

        return history[-limit:]

    def get_speed_history(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return simplified network speed history.
        """

        history = self.get_history(
            limit=limit
        )

        result = []

        for item in history:

            speed = item.get(
                "speed",
                {},
            )

            result.append(
                {
                    "timestamp": item.get(
                        "timestamp"
                    ),
                    "download_mbps": (
                        speed.get(
                            "download_mbps"
                        )
                        if isinstance(
                            speed,
                            dict,
                        )
                        else None
                    ),
                    "upload_mbps": (
                        speed.get(
                            "upload_mbps"
                        )
                        if isinstance(
                            speed,
                            dict,
                        )
                        else None
                    ),
                }
            )

        return result

    def clear_history(
        self,
    ) -> None:

        self.history.clear()

    # ============================================================
    # RESET
    # ============================================================

    def reset(
        self,
    ) -> None:
        """
        Reset network monitor state.
        """

        self.history.clear()

        self.last_stats = None

        self.last_update_time = None

        self._previous_io = None

        self._previous_io_time = None


__all__ = [
    "NetworkMonitor",
]


