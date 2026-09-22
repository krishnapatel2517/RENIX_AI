"""
RENIX Sensor Manager
====================

Central sensor management system for RENIX robotics.

Features:
- Register multiple sensors
- Connect and disconnect sensors
- Read sensor data
- Continuous sensor monitoring
- Sensor history
- Threshold alerts
- Event callbacks
- Sensor health monitoring
- Support for hardware adapters
- Custom sensor types

Supported examples:
- Ultrasonic distance sensors
- IR sensors
- Temperature sensors
- Humidity sensors
- Motion sensors
- Light sensors
- Gas sensors
- Pressure sensors
- IMU / Gyroscope
- Accelerometer
- GPS
- Custom sensors
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(
    "RENIX.Robotics.SensorManager"
)


class SensorManager:
    """
    Central manager for robotic sensors.

    Each sensor can have its own hardware adapter
    or use a shared adapter.
    """

    def __init__(
        self,
        adapter: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize SensorManager.

        Args:
            adapter:
                Shared hardware adapter.

            config:
                Sensor manager configuration.
        """

        self.adapter = adapter
        self.config = config or {}

        self.sensors: Dict[
            str,
            Dict[str, Any]
        ] = {}

        self.connected = False

        self.monitoring = False

        self.monitor_thread: Optional[
            threading.Thread
        ] = None

        self.monitor_interval = float(
            self.config.get(
                "monitor_interval",
                0.5,
            )
        )

        self.max_history = int(
            self.config.get(
                "max_history",
                500,
            )
        )

        self.event_handlers: List[
            Callable[[Dict[str, Any]], None]
        ] = []

        self._stop_event = threading.Event()

        self._lock = threading.RLock()

        logger.info(
            "SensorManager initialized."
        )

    # ============================================================
    # CONNECTION
    # ============================================================

    def connect(self) -> bool:
        """
        Connect to the sensor hardware.
        """

        if self.connected:
            return True

        try:

            if (
                self.adapter is not None
                and hasattr(
                    self.adapter,
                    "connect",
                )
            ):

                result = (
                    self.adapter.connect()
                )

                if result is False:

                    return False

            self.connected = True

            self._emit_event(
                "sensor_manager_connected"
            )

            logger.info(
                "Sensor manager connected."
            )

            return True

        except Exception as error:

            logger.error(
                "Failed to connect sensor manager: %s",
                error,
            )

            return False

    def disconnect(self) -> bool:
        """
        Disconnect sensor hardware.
        """

        self.stop_monitoring()

        try:

            if (
                self.adapter is not None
                and hasattr(
                    self.adapter,
                    "disconnect",
                )
            ):

                result = (
                    self.adapter.disconnect()
                )

                if result is False:

                    return False

            self.connected = False

            self._emit_event(
                "sensor_manager_disconnected"
            )

            return True

        except Exception as error:

            logger.error(
                "Failed to disconnect sensor manager: %s",
                error,
            )

            return False

    # ============================================================
    # SENSOR REGISTRATION
    # ============================================================

    def register_sensor(
        self,
        sensor_id: str,
        sensor_type: str,
        config: Optional[
            Dict[str, Any]
        ] = None,
        reader: Optional[
            Callable[[], Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Register a sensor.

        Args:
            sensor_id:
                Unique sensor identifier.

            sensor_type:
                Type of sensor.

            config:
                Sensor configuration.

            reader:
                Optional custom function used
                to read sensor data.

        Example:

            register_sensor(
                sensor_id="front_distance",
                sensor_type="ultrasonic",
                config={
                    "trigger_pin": 5,
                    "echo_pin": 6,
                    "min_threshold": 5,
                    "max_threshold": 200,
                }
            )
        """

        sensor_id = (
            str(sensor_id)
            .strip()
            .lower()
            .replace(
                " ",
                "_",
            )
        )

        sensor_type = (
            str(sensor_type)
            .strip()
            .lower()
        )

        sensor_config = config or {}

        with self._lock:

            if sensor_id in self.sensors:

                return {
                    "success": False,
                    "error": (
                        f"Sensor '{sensor_id}' "
                        "already exists."
                    ),
                }

            self.sensors[sensor_id] = {
                "sensor_id": sensor_id,
                "sensor_type": sensor_type,
                "config": sensor_config,
                "reader": reader,
                "enabled": True,
                "connected": False,
                "last_value": None,
                "last_read_time": None,
                "read_count": 0,
                "error_count": 0,
                "history": deque(
                    maxlen=self.max_history
                ),
            }

        self._emit_event(
            "sensor_registered",
            sensor_id=sensor_id,
            sensor_type=sensor_type,
        )

        logger.info(
            "Sensor registered: %s (%s)",
            sensor_id,
            sensor_type,
        )

        return {
            "success": True,
            "sensor": self.get_sensor(
                sensor_id
            ),
        }

    def unregister_sensor(
        self,
        sensor_id: str,
    ) -> bool:
        """
        Remove a sensor.
        """

        sensor_id = self._normalize_id(
            sensor_id
        )

        self.disconnect_sensor(
            sensor_id
        )

        with self._lock:

            if sensor_id not in self.sensors:

                return False

            del self.sensors[sensor_id]

        self._emit_event(
            "sensor_unregistered",
            sensor_id=sensor_id,
        )

        return True

    # ============================================================
    # SENSOR CONNECTION
    # ============================================================

    def connect_sensor(
        self,
        sensor_id: str,
    ) -> Dict[str, Any]:
        """
        Connect an individual sensor.
        """

        sensor = self._get_sensor_reference(
            sensor_id
        )

        if sensor is None:

            return {
                "success": False,
                "error": "Sensor not found.",
            }

        try:

            if (
                self.adapter is not None
                and hasattr(
                    self.adapter,
                    "connect_sensor",
                )
            ):

                result = (
                    self.adapter.connect_sensor(
                        sensor["sensor_id"],
                        sensor["config"],
                    )
                )

                if result is False:

                    return {
                        "success": False,
                        "error": (
                            "Failed to connect sensor."
                        ),
                    }

            sensor["connected"] = True

            return {
                "success": True,
                "sensor_id": sensor[
                    "sensor_id"
                ],
            }

        except Exception as error:

            sensor["error_count"] += 1

            return {
                "success": False,
                "error": str(error),
            }

    def disconnect_sensor(
        self,
        sensor_id: str,
    ) -> Dict[str, Any]:
        """
        Disconnect an individual sensor.
        """

        sensor = self._get_sensor_reference(
            sensor_id
        )

        if sensor is None:

            return {
                "success": False,
                "error": "Sensor not found.",
            }

        try:

            if (
                self.adapter is not None
                and hasattr(
                    self.adapter,
                    "disconnect_sensor",
                )
            ):

                self.adapter.disconnect_sensor(
                    sensor["sensor_id"]
                )

            sensor["connected"] = False

            return {
                "success": True,
                "sensor_id": sensor[
                    "sensor_id"
                ],
            }

        except Exception as error:

            return {
                "success": False,
                "error": str(error),
            }

    # ============================================================
    # SENSOR READING
    # ============================================================

    def read_sensor(
        self,
        sensor_id: str,
    ) -> Dict[str, Any]:
        """
        Read data from a sensor.
        """

        sensor = self._get_sensor_reference(
            sensor_id
        )

        if sensor is None:

            return {
                "success": False,
                "error": "Sensor not found.",
            }

        if not sensor["enabled"]:

            return {
                "success": False,
                "error": "Sensor is disabled.",
            }

        try:

            value = self._read_value(
                sensor
            )

            timestamp = (
                datetime.now()
                .isoformat()
            )

            reading = {
                "sensor_id": sensor[
                    "sensor_id"
                ],
                "sensor_type": sensor[
                    "sensor_type"
                ],
                "value": value,
                "timestamp": timestamp,
            }

            with self._lock:

                sensor["last_value"] = value

                sensor[
                    "last_read_time"
                ] = timestamp

                sensor[
                    "read_count"
                ] += 1

                sensor[
                    "history"
                ].append(
                    reading
                )

            self._check_thresholds(
                sensor,
                value,
            )

            self._emit_event(
                "sensor_reading",
                **reading,
            )

            return {
                "success": True,
                **reading,
            }

        except Exception as error:

            with self._lock:

                sensor[
                    "error_count"
                ] += 1

            logger.error(
                "Sensor read failed (%s): %s",
                sensor_id,
                error,
            )

            self._emit_event(
                "sensor_error",
                sensor_id=sensor_id,
                error=str(error),
            )

            return {
                "success": False,
                "sensor_id": sensor_id,
                "error": str(error),
            }

    def read_all(
        self,
    ) -> Dict[str, Any]:
        """
        Read all enabled sensors.
        """

        results = {}

        sensor_ids = list(
            self.sensors.keys()
        )

        for sensor_id in sensor_ids:

            results[sensor_id] = (
                self.read_sensor(
                    sensor_id
                )
            )

        success = all(
            result.get(
                "success",
                False,
            )
            for result in results.values()
        )

        return {
            "success": success,
            "results": results,
        }

    # ============================================================
    # SENSOR MONITORING
    # ============================================================

    def start_monitoring(
        self,
        interval: Optional[
            float
        ] = None,
    ) -> bool:
        """
        Start continuous sensor monitoring.
        """

        if self.monitoring:

            return True

        if interval is not None:

            self.monitor_interval = max(
                0.05,
                float(interval),
            )

        self._stop_event.clear()

        self.monitoring = True

        self.monitor_thread = (
            threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name=(
                    "RENIX-SensorMonitor"
                ),
            )
        )

        self.monitor_thread.start()

        self._emit_event(
            "sensor_monitoring_started"
        )

        logger.info(
            "Sensor monitoring started."
        )

        return True

    def stop_monitoring(
        self,
    ) -> None:
        """
        Stop continuous monitoring.
        """

        if not self.monitoring:

            return

        self.monitoring = False

        self._stop_event.set()

        if (
            self.monitor_thread
            and self.monitor_thread.is_alive()
        ):

            self.monitor_thread.join(
                timeout=2
            )

        self.monitor_thread = None

        self._emit_event(
            "sensor_monitoring_stopped"
        )

        logger.info(
            "Sensor monitoring stopped."
        )

    def _monitor_loop(
        self,
    ) -> None:
        """
        Background monitoring loop.
        """

        while (
            not self._stop_event.is_set()
        ):

            try:

                self.read_all()

            except Exception as error:

                logger.error(
                    "Sensor monitor error: %s",
                    error,
                )

            self._stop_event.wait(
                self.monitor_interval
            )

    # ============================================================
    # SENSOR ENABLE / DISABLE
    # ============================================================

    def enable_sensor(
        self,
        sensor_id: str,
    ) -> bool:
        """
        Enable a sensor.
        """

        sensor = self._get_sensor_reference(
            sensor_id
        )

        if sensor is None:

            return False

        sensor["enabled"] = True

        self._emit_event(
            "sensor_enabled",
            sensor_id=sensor_id,
        )

        return True

    def disable_sensor(
        self,
        sensor_id: str,
    ) -> bool:
        """
        Disable a sensor.
        """

        sensor = self._get_sensor_reference(
            sensor_id
        )

        if sensor is None:

            return False

        sensor["enabled"] = False

        self._emit_event(
            "sensor_disabled",
            sensor_id=sensor_id,
        )

        return True

    # ============================================================
    # THRESHOLDS
    # ============================================================

    def set_thresholds(
        self,
        sensor_id: str,
        minimum: Optional[
            float
        ] = None,
        maximum: Optional[
            float
        ] = None,
    ) -> bool:
        """
        Configure sensor thresholds.
        """

        sensor = self._get_sensor_reference(
            sensor_id
        )

        if sensor is None:

            return False

        sensor["config"][
            "min_threshold"
        ] = minimum

        sensor["config"][
            "max_threshold"
        ] = maximum

        return True

    def _check_thresholds(
        self,
        sensor: Dict[str, Any],
        value: Any,
    ) -> None:
        """
        Check sensor value against thresholds.
        """

        if not isinstance(
            value,
            (
                int,
                float,
            ),
        ):
            return

        config = sensor[
            "config"
        ]

        minimum = config.get(
            "min_threshold"
        )

        maximum = config.get(
            "max_threshold"
        )

        if (
            minimum is not None
            and value < minimum
        ):

            self._emit_event(
                "sensor_threshold_low",
                sensor_id=sensor[
                    "sensor_id"
                ],
                value=value,
                threshold=minimum,
            )

        if (
            maximum is not None
            and value > maximum
        ):

            self._emit_event(
                "sensor_threshold_high",
                sensor_id=sensor[
                    "sensor_id"
                ],
                value=value,
                threshold=maximum,
            )

    # ============================================================
    # SENSOR HISTORY
    # ============================================================

    def get_history(
        self,
        sensor_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Return sensor reading history.
        """

        sensor = self._get_sensor_reference(
            sensor_id
        )

        if sensor is None:

            return []

        history = list(
            sensor["history"]
        )

        return history[
            -max(
                1,
                limit,
            ):
        ]

    def clear_history(
        self,
        sensor_id: Optional[
            str
        ] = None,
    ) -> None:
        """
        Clear sensor history.

        If sensor_id is None,
        clears all sensor histories.
        """

        if sensor_id is not None:

            sensor = (
                self._get_sensor_reference(
                    sensor_id
                )
            )

            if sensor:

                sensor["history"].clear()

            return

        for sensor in (
            self.sensors.values()
        ):

            sensor["history"].clear()

    # ============================================================
    # SENSOR INFORMATION
    # ============================================================

    def get_sensor(
        self,
        sensor_id: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Return sensor information.
        """

        sensor = self._get_sensor_reference(
            sensor_id
        )

        if sensor is None:

            return None

        data = {}

        for key, value in (
            sensor.items()
        ):

            if key in (
                "reader",
                "history",
            ):

                continue

            data[key] = value

        data["history_size"] = len(
            sensor["history"]
        )

        return data

    def get_all_sensors(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return all registered sensors.
        """

        return {
            sensor_id: self.get_sensor(
                sensor_id
            )
            for sensor_id in (
                self.sensors.keys()
            )
        }

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return complete sensor manager status.
        """

        return {
            "connected": self.connected,
            "monitoring": self.monitoring,
            "sensor_count": len(
                self.sensors
            ),
            "monitor_interval": (
                self.monitor_interval
            ),
            "sensors": (
                self.get_all_sensors()
            ),
        }

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:
        """
        Perform health check for all sensors.
        """

        results = {}

        healthy_count = 0

        for sensor_id, sensor in (
            self.sensors.items()
        ):

            healthy = (
                sensor["enabled"]
                and sensor[
                    "error_count"
                ] < 10
            )

            results[sensor_id] = {
                "healthy": healthy,
                "read_count": sensor[
                    "read_count"
                ],
                "error_count": sensor[
                    "error_count"
                ],
                "last_read_time": sensor[
                    "last_read_time"
                ],
            }

            if healthy:

                healthy_count += 1

        return {
            "healthy": (
                healthy_count
                == len(self.sensors)
                if self.sensors
                else True
            ),
            "healthy_sensors": healthy_count,
            "total_sensors": len(
                self.sensors
            ),
            "sensors": results,
        }

    # ============================================================
    # EVENTS
    # ============================================================

    def add_event_handler(
        self,
        handler: Callable[
            [Dict[str, Any]],
            None,
        ],
    ) -> bool:
        """
        Add sensor event handler.
        """

        if not callable(handler):

            return False

        with self._lock:

            if (
                handler
                not in self.event_handlers
            ):

                self.event_handlers.append(
                    handler
                )

        return True

    def remove_event_handler(
        self,
        handler: Callable,
    ) -> bool:
        """
        Remove sensor event handler.
        """

        with self._lock:

            if (
                handler
                not in self.event_handlers
            ):

                return False

            self.event_handlers.remove(
                handler
            )

        return True

    def _emit_event(
        self,
        event_name: str,
        **data: Any,
    ) -> None:
        """
        Emit sensor event.
        """

        event = {
            "event": event_name,
            "timestamp": (
                datetime.now()
                .isoformat()
            ),
            **data,
        }

        handlers = list(
            self.event_handlers
        )

        for handler in handlers:

            try:

                handler(
                    event.copy()
                )

            except Exception as error:

                logger.error(
                    "Sensor event handler failed: %s",
                    error,
                )

    # ============================================================
    # HARDWARE READING
    # ============================================================

    def _read_value(
        self,
        sensor: Dict[str, Any],
    ) -> Any:
        """
        Read value using the configured reader
        or hardware adapter.
        """

        reader = sensor.get(
            "reader"
        )

        if reader is not None:

            return reader()

        if (
            self.adapter is not None
            and hasattr(
                self.adapter,
                "read_sensor",
            )
        ):

            return self.adapter.read_sensor(
                sensor["sensor_id"]
            )

        raise RuntimeError(
            "No sensor reader configured."
        )

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _normalize_id(
        sensor_id: str,
    ) -> str:
        """
        Normalize sensor ID.
        """

        return (
            str(sensor_id)
            .strip()
            .lower()
            .replace(
                " ",
                "_",
            )
        )

    def _get_sensor_reference(
        self,
        sensor_id: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Get internal sensor reference.
        """

        sensor_id = (
            self._normalize_id(
                sensor_id
            )
        )

        with self._lock:

            return self.sensors.get(
                sensor_id
            )

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Safely shutdown SensorManager.
        """

        logger.info(
            "Shutting down SensorManager..."
        )

        self.stop_monitoring()

        self.disconnect()

        with self._lock:

            self.event_handlers.clear()

        logger.info(
            "SensorManager shutdown complete."
        )


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )


    class DemoSensorAdapter:

        def connect(self):

            print(
                "Sensor hardware connected."
            )

            return True

        def disconnect(self):

            print(
                "Sensor hardware disconnected."
            )

            return True


    def fake_distance_sensor():

        return 42.5


    def fake_temperature_sensor():

        return 28.7


    manager = SensorManager(
        adapter=DemoSensorAdapter(),
        config={
            "monitor_interval": 1,
            "max_history": 100,
        },
    )

    manager.connect()

    manager.register_sensor(
        sensor_id="front_distance",
        sensor_type="ultrasonic",
        config={
            "min_threshold": 5,
            "max_threshold": 100,
        },
        reader=fake_distance_sensor,
    )

    manager.register_sensor(
        sensor_id="temperature",
        sensor_type="temperature",
        config={
            "max_threshold": 50,
        },
        reader=fake_temperature_sensor,
    )

    print("\nSingle sensor reading:")
    print(
        manager.read_sensor(
            "front_distance"
        )
    )

    print("\nAll sensors:")
    print(
        manager.read_all()
    )

    print("\nSensor status:")
    print(
        manager.get_status()
    )

    manager.shutdown()


