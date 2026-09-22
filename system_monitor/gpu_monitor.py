"""
RENIX GPU Monitor
=================

Provides GPU monitoring for RENIX.

Features:
- NVIDIA GPU monitoring through pynvml when available
- GPU name and driver information
- GPU utilization
- VRAM usage
- GPU temperature
- GPU power usage
- Fan speed
- Multi-GPU support
- Monitoring history
- Basic GPU health analysis

Optional dependency:
    pip install nvidia-ml-py
"""

from __future__ import annotations

import platform
import subprocess
import time
from collections import deque
from datetime import datetime
from typing import Any

try:
    import pynvml
except ImportError:
    pynvml = None


class GPUMonitor:
    """
    GPU monitoring component for RENIX.

    Supports NVIDIA GPUs through NVML.

    Example:
        monitor = GPUMonitor()

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

        self._nvml_initialized = False

        self._initialize_nvml()

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def _initialize_nvml(
        self,
    ) -> None:
        """
        Initialize NVIDIA Management Library.
        """

        if pynvml is None:
            return

        try:

            pynvml.nvmlInit()

            self._nvml_initialized = True

        except Exception:

            self._nvml_initialized = False

    def shutdown(
        self,
    ) -> None:
        """
        Shut down NVML safely.
        """

        if (
            pynvml is None
            or not self._nvml_initialized
        ):
            return

        try:

            pynvml.nvmlShutdown()

        except Exception:
            pass

        self._nvml_initialized = False

    # ============================================================
    # MAIN API
    # ============================================================

    def get_stats(
        self,
        *,
        store: bool = True,
    ) -> dict[str, Any]:
        """
        Return complete GPU statistics.
        """

        gpus = self._get_all_gpus()

        available = len(gpus) > 0

        stats = {
            "available": available,
            "timestamp": (
                datetime.now().isoformat(
                    timespec="seconds"
                )
            ),
            "gpu_count": len(gpus),
            "gpus": gpus,
            "platform": platform.system(),
        }

        if available:

            stats.update(
                self._create_summary(
                    gpus
                )
            )

        else:

            stats[
                "message"
            ] = (
                "No supported GPU monitoring "
                "backend available."
            )

        self.last_stats = stats
        self.last_update_time = (
            time.time()
        )

        if store:
            self.history.append(
                stats
            )

        return stats

    # ============================================================
    # GPU DISCOVERY
    # ============================================================

    def _get_all_gpus(
        self,
    ) -> list[dict[str, Any]]:
        """
        Detect and collect information
        about all available GPUs.
        """

        gpus: list[
            dict[str, Any]
        ] = []

        if self._nvml_initialized:

            try:

                count = (
                    pynvml.nvmlDeviceGetCount()
                )

                for index in range(
                    count
                ):

                    gpu = (
                        self._get_nvidia_gpu(
                            index
                        )
                    )

                    if gpu:
                        gpus.append(
                            gpu
                        )

                return gpus

            except Exception:
                pass

        fallback_gpus = (
            self._detect_windows_gpu()
        )

        return fallback_gpus

    # ============================================================
    # NVIDIA GPU
    # ============================================================

    def _get_nvidia_gpu(
        self,
        index: int,
    ) -> dict[str, Any] | None:
        """
        Collect detailed statistics
        for one NVIDIA GPU.
        """

        if not self._nvml_initialized:
            return None

        try:

            handle = (
                pynvml.nvmlDeviceGetHandleByIndex(
                    index
                )
            )

            name = (
                pynvml.nvmlDeviceGetName(
                    handle
                )
            )

            if isinstance(
                name,
                bytes,
            ):
                name = name.decode(
                    errors="replace"
                )

            memory = (
                pynvml.nvmlDeviceGetMemoryInfo(
                    handle
                )
            )

            utilization = (
                pynvml.nvmlDeviceGetUtilizationRates(
                    handle
                )
            )

            temperature = (
                self._safe_nvml_call(
                    pynvml.nvmlDeviceGetTemperature,
                    handle,
                    pynvml.NVML_TEMPERATURE_GPU,
                )
            )

            fan_speed = (
                self._safe_nvml_call(
                    pynvml.nvmlDeviceGetFanSpeed,
                    handle,
                )
            )

            power_usage = (
                self._safe_nvml_call(
                    pynvml.nvmlDeviceGetPowerUsage,
                    handle,
                )
            )

            power_limit = (
                self._safe_nvml_call(
                    pynvml.nvmlDeviceGetEnforcedPowerLimit,
                    handle,
                )
            )

            driver_version = (
                self._get_driver_version()
            )

            memory_total = (
                int(memory.total)
            )

            memory_used = (
                int(memory.used)
            )

            memory_free = (
                int(memory.free)
            )

            memory_percent = (
                round(
                    (
                        memory_used
                        / memory_total
                        * 100
                    ),
                    2,
                )
                if memory_total > 0
                else 0.0
            )

            return {
                "index": index,
                "vendor": "NVIDIA",
                "name": str(name),
                "driver_version": (
                    driver_version
                ),
                "utilization_percent": (
                    float(
                        utilization.gpu
                    )
                ),
                "memory_utilization_percent": (
                    float(
                        utilization.memory
                    )
                ),
                "memory": {
                    "total_bytes": (
                        memory_total
                    ),
                    "used_bytes": (
                        memory_used
                    ),
                    "free_bytes": (
                        memory_free
                    ),
                    "percent": (
                        memory_percent
                    ),
                    "total_mb": (
                        round(
                            memory_total
                            / 1024
                            / 1024,
                            2,
                        )
                    ),
                    "used_mb": (
                        round(
                            memory_used
                            / 1024
                            / 1024,
                            2,
                        )
                    ),
                    "free_mb": (
                        round(
                            memory_free
                            / 1024
                            / 1024,
                            2,
                        )
                    ),
                },
                "temperature_celsius": (
                    temperature
                ),
                "fan_speed_percent": (
                    fan_speed
                ),
                "power_usage_watts": (
                    round(
                        power_usage
                        / 1000,
                        2,
                    )
                    if power_usage
                    is not None
                    else None
                ),
                "power_limit_watts": (
                    round(
                        power_limit
                        / 1000,
                        2,
                    )
                    if power_limit
                    is not None
                    else None
                ),
            }

        except Exception:
            return None

    # ============================================================
    # NVML UTILITY
    # ============================================================

    @staticmethod
    def _safe_nvml_call(
        function: Any,
        *args: Any,
    ) -> Any:
        """
        Safely call an NVML function.
        """

        try:
            return function(
                *args
            )

        except Exception:
            return None

    def _get_driver_version(
        self,
    ) -> str | None:

        if not self._nvml_initialized:
            return None

        try:

            version = (
                pynvml.nvmlSystemGetDriverVersion()
            )

            if isinstance(
                version,
                bytes,
            ):
                return version.decode(
                    errors="replace"
                )

            return str(version)

        except Exception:
            return None

    # ============================================================
    # WINDOWS FALLBACK DETECTION
    # ============================================================

    def _detect_windows_gpu(
        self,
    ) -> list[dict[str, Any]]:
        """
        Basic GPU detection fallback.

        This detects GPU names when detailed
        telemetry is unavailable.
        """

        if platform.system() != "Windows":
            return []

        try:

            command = [
                "wmic",
                "path",
                "win32_VideoController",
                "get",
                "Name",
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

            lines = (
                result.stdout
                .splitlines()
            )

            names = []

            for line in lines[1:]:

                name = line.strip()

                if (
                    name
                    and name not in names
                ):

                    names.append(
                        name
                    )

            gpus = []

            for index, name in enumerate(
                names
            ):

                vendor = (
                    self._detect_vendor(
                        name
                    )
                )

                gpus.append(
                    {
                        "index": index,
                        "vendor": vendor,
                        "name": name,
                        "utilization_percent": None,
                        "memory_utilization_percent": None,
                        "memory": {},
                        "temperature_celsius": None,
                        "fan_speed_percent": None,
                        "power_usage_watts": None,
                        "telemetry_available": False,
                    }
                )

            return gpus

        except Exception:
            return []

    @staticmethod
    def _detect_vendor(
        name: str,
    ) -> str:

        name_lower = name.lower()

        if "nvidia" in name_lower:
            return "NVIDIA"

        if (
            "amd" in name_lower
            or "radeon" in name_lower
        ):
            return "AMD"

        if "intel" in name_lower:
            return "Intel"

        return "Unknown"

    # ============================================================
    # SUMMARY
    # ============================================================

    def _create_summary(
        self,
        gpus: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Create a simplified GPU summary.
        """

        utilization_values = []
        temperatures = []
        memory_used = 0
        memory_total = 0

        for gpu in gpus:

            utilization = gpu.get(
                "utilization_percent"
            )

            if isinstance(
                utilization,
                (int, float),
            ):
                utilization_values.append(
                    float(utilization)
                )

            temperature = gpu.get(
                "temperature_celsius"
            )

            if isinstance(
                temperature,
                (int, float),
            ):
                temperatures.append(
                    float(temperature)
                )

            memory = gpu.get(
                "memory",
                {},
            )

            if isinstance(
                memory,
                dict,
            ):

                memory_used += (
                    int(
                        memory.get(
                            "used_bytes",
                            0,
                        )
                        or 0
                    )
                )

                memory_total += (
                    int(
                        memory.get(
                            "total_bytes",
                            0,
                        )
                        or 0
                    )
                )

        return {
            "average_utilization_percent": (
                round(
                    sum(
                        utilization_values
                    )
                    / len(
                        utilization_values
                    ),
                    2,
                )
                if utilization_values
                else None
            ),
            "highest_temperature_celsius": (
                max(temperatures)
                if temperatures
                else None
            ),
            "total_vram_bytes": (
                memory_total
            ),
            "used_vram_bytes": (
                memory_used
            ),
            "vram_usage_percent": (
                round(
                    memory_used
                    / memory_total
                    * 100,
                    2,
                )
                if memory_total > 0
                else None
            ),
        }

    # ============================================================
    # INDIVIDUAL GPU ACCESS
    # ============================================================

    def get_gpu(
        self,
        index: int = 0,
    ) -> dict[str, Any] | None:
        """
        Return information about one GPU.
        """

        stats = self.get_stats(
            store=False
        )

        gpus = stats.get(
            "gpus",
            [],
        )

        if (
            not isinstance(
                gpus,
                list,
            )
            or index < 0
            or index >= len(gpus)
        ):
            return None

        return gpus[index]

    def get_gpu_count(
        self,
    ) -> int:

        stats = self.get_stats(
            store=False
        )

        return int(
            stats.get(
                "gpu_count",
                0,
            )
        )

    # ============================================================
    # QUICK METRICS
    # ============================================================

    def get_usage(
        self,
        index: int = 0,
    ) -> float | None:

        gpu = self.get_gpu(
            index
        )

        if gpu is None:
            return None

        usage = gpu.get(
            "utilization_percent"
        )

        return (
            float(usage)
            if isinstance(
                usage,
                (int, float),
            )
            else None
        )

    def get_temperature(
        self,
        index: int = 0,
    ) -> float | None:

        gpu = self.get_gpu(
            index
        )

        if gpu is None:
            return None

        temperature = gpu.get(
            "temperature_celsius"
        )

        return (
            float(temperature)
            if isinstance(
                temperature,
                (int, float),
            )
            else None
        )

    def get_memory_usage(
        self,
        index: int = 0,
    ) -> dict[str, Any]:

        gpu = self.get_gpu(
            index
        )

        if gpu is None:
            return {}

        memory = gpu.get(
            "memory",
            {},
        )

        return (
            memory
            if isinstance(
                memory,
                dict,
            )
            else {}
        )

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
        index: int = 0,
    ) -> dict[str, Any]:
        """
        Return simplified GPU health status.
        """

        gpu = self.get_gpu(
            index
        )

        if gpu is None:

            return {
                "available": False,
                "status": "unknown",
            }

        usage = gpu.get(
            "utilization_percent"
        )

        temperature = gpu.get(
            "temperature_celsius"
        )

        memory = gpu.get(
            "memory",
            {},
        )

        memory_percent = (
            memory.get(
                "percent"
            )
            if isinstance(
                memory,
                dict,
            )
            else None
        )

        issues = []

        if (
            isinstance(
                temperature,
                (int, float),
            )
            and temperature >= 90
        ):
            issues.append(
                "GPU temperature is critical."
            )

        elif (
            isinstance(
                temperature,
                (int, float),
            )
            and temperature >= 80
        ):
            issues.append(
                "GPU temperature is high."
            )

        if (
            isinstance(
                memory_percent,
                (int, float),
            )
            and memory_percent >= 95
        ):
            issues.append(
                "VRAM usage is critical."
            )

        if (
            isinstance(
                usage,
                (int, float),
            )
            and usage >= 95
        ):
            issues.append(
                "GPU utilization is very high."
            )

        if any(
            "critical"
            in issue.lower()
            for issue in issues
        ):
            status = "critical"

        elif issues:
            status = "warning"

        else:
            status = "normal"

        return {
            "available": True,
            "status": status,
            "healthy": (
                status == "normal"
            ),
            "issues": issues,
            "usage_percent": usage,
            "temperature_celsius": (
                temperature
            ),
            "memory_percent": (
                memory_percent
            ),
        }

    # ============================================================
    # ANALYSIS
    # ============================================================

    def analyze(
        self,
    ) -> dict[str, Any]:
        """
        Analyze all detected GPUs.
        """

        stats = self.get_stats()

        if not stats.get(
            "available"
        ):

            return {
                "available": False,
                "status": "unknown",
                "issues": [
                    "No supported GPU telemetry available."
                ],
                "recommendations": [],
            }

        analyses = []

        for index in range(
            stats.get(
                "gpu_count",
                0,
            )
        ):

            analyses.append(
                {
                    "gpu_index": index,
                    "analysis": (
                        self.get_status(
                            index
                        )
                    ),
                }
            )

        critical = any(
            item[
                "analysis"
            ].get(
                "status"
            )
            == "critical"
            for item in analyses
        )

        warning = any(
            item[
                "analysis"
            ].get(
                "status"
            )
            == "warning"
            for item in analyses
        )

        if critical:
            status = "critical"

        elif warning:
            status = "warning"

        else:
            status = "normal"

        recommendations = []

        if critical:

            recommendations.append(
                "Reduce GPU workload and check cooling."
            )

        if warning:

            recommendations.append(
                "Monitor temperatures and VRAM usage."
            )

        return {
            "available": True,
            "status": status,
            "gpu_count": stats.get(
                "gpu_count"
            ),
            "analyses": analyses,
            "recommendations": (
                recommendations
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

    def get_usage_history(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return simplified GPU usage history.
        """

        history = self.get_history(
            limit=limit
        )

        result = []

        for item in history:

            result.append(
                {
                    "timestamp": item.get(
                        "timestamp"
                    ),
                    "average_utilization_percent": (
                        item.get(
                            "average_utilization_percent"
                        )
                    ),
                    "highest_temperature_celsius": (
                        item.get(
                            "highest_temperature_celsius"
                        )
                    ),
                    "vram_usage_percent": (
                        item.get(
                            "vram_usage_percent"
                        )
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
        Reset monitor state.
        """

        self.history.clear()

        self.last_stats = None

        self.last_update_time = None

    # ============================================================
    # CLEANUP
    # ============================================================

    def __del__(
        self,
    ) -> None:

        try:
            self.shutdown()

        except Exception:
            pass


__all__ = [
    "GPUMonitor",
]


