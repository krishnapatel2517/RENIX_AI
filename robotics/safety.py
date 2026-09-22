"""
RENIX Robotics Safety System
============================

Provides safety controls for RENIX robotics.

Features:
- Emergency stop
- Obstacle safety
- Distance safety
- Motor safety
- Servo safety
- Battery safety
- Temperature safety
- Sensor health monitoring
- Movement permission checks
- Safety zones
- Speed limits
- Automatic shutdown
- Safety event logging
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger("RENIX.Robotics.Safety")


class RobotSafety:
    """
    Central safety controller for RENIX robotics.

    This class monitors robot safety conditions and can
    stop the robot when dangerous conditions are detected.

    It can integrate with:
    - RobotController
    - ServoController
    - SensorManager
    - Navigation
    - ObjectManipulation
    """

    def __init__(
        self,
        robot_controller: Optional[Any] = None,
        servo_controller: Optional[Any] = None,
        sensor_manager: Optional[Any] = None,
        navigation: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.robot_controller = robot_controller
        self.servo_controller = servo_controller
        self.sensor_manager = sensor_manager
        self.navigation = navigation

        self.config = config or {}

        # --------------------------------------------------------
        # Safety state
        # --------------------------------------------------------

        self.enabled = True
        self.monitoring = False

        self.emergency_active = False
        self.emergency_reason: Optional[str] = None

        self.last_emergency_time: Optional[str] = None

        # --------------------------------------------------------
        # Safety limits
        # --------------------------------------------------------

        self.min_obstacle_distance = float(
            self.config.get(
                "min_obstacle_distance",
                20.0,
            )
        )

        self.warning_obstacle_distance = float(
            self.config.get(
                "warning_obstacle_distance",
                40.0,
            )
        )

        self.max_temperature = float(
            self.config.get(
                "max_temperature",
                80.0,
            )
        )

        self.max_motor_temperature = float(
            self.config.get(
                "max_motor_temperature",
                75.0,
            )
        )

        self.low_battery_level = float(
            self.config.get(
                "low_battery_level",
                20.0,
            )
        )

        self.critical_battery_level = float(
            self.config.get(
                "critical_battery_level",
                10.0,
            )
        )

        self.max_speed = float(
            self.config.get(
                "max_speed",
                100.0,
            )
        )

        self.monitor_interval = float(
            self.config.get(
                "monitor_interval",
                0.25,
            )
        )

        # --------------------------------------------------------
        # Safety zones
        # --------------------------------------------------------

        self.safety_zones: List[
            Dict[str, Any]
        ] = []

        # --------------------------------------------------------
        # Event system
        # --------------------------------------------------------

        self.event_handlers: List[
            Callable[[Dict[str, Any]], None]
        ] = []

        self.safety_history: List[
            Dict[str, Any]
        ] = []

        self.max_history = int(
            self.config.get(
                "max_history",
                1000,
            )
        )

        # --------------------------------------------------------
        # Thread control
        # --------------------------------------------------------

        self._lock = threading.RLock()

        self._stop_event = threading.Event()

        self._monitor_thread: Optional[
            threading.Thread
        ] = None

        logger.info(
            "RobotSafety initialized."
        )

    # ============================================================
    # SAFETY ENABLE / DISABLE
    # ============================================================

    def enable(self) -> bool:
        """
        Enable robotics safety system.
        """

        with self._lock:

            self.enabled = True

        self._emit_event(
            "safety_enabled"
        )

        logger.info(
            "Robot safety enabled."
        )

        return True

    def disable(self) -> bool:
        """
        Disable safety monitoring.

        WARNING:
        Emergency safety state cannot be disabled
        while active.
        """

        if self.emergency_active:

            logger.warning(
                "Cannot disable safety during emergency."
            )

            return False

        with self._lock:

            self.enabled = False

        self.stop_monitoring()

        self._emit_event(
            "safety_disabled"
        )

        logger.warning(
            "Robot safety disabled."
        )

        return True

    # ============================================================
    # MONITORING
    # ============================================================

    def start_monitoring(self) -> bool:
        """
        Start background safety monitoring.
        """

        if self.monitoring:
            return True

        self.enabled = True
        self.monitoring = True

        self._stop_event.clear()

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="RENIX-RobotSafety",
        )

        self._monitor_thread.start()

        self._emit_event(
            "safety_monitoring_started"
        )

        logger.info(
            "Robot safety monitoring started."
        )

        return True

    def stop_monitoring(self) -> None:
        """
        Stop background safety monitoring.
        """

        if not self.monitoring:
            return

        self.monitoring = False

        self._stop_event.set()

        if (
            self._monitor_thread
            and self._monitor_thread.is_alive()
        ):

            self._monitor_thread.join(
                timeout=2
            )

        self._monitor_thread = None

        self._emit_event(
            "safety_monitoring_stopped"
        )

        logger.info(
            "Robot safety monitoring stopped."
        )

    def _monitor_loop(self) -> None:
        """
        Main background safety monitoring loop.
        """

        while (
            self.monitoring
            and not self._stop_event.is_set()
        ):

            try:

                if self.enabled:

                    self.perform_safety_check()

            except Exception as error:

                logger.error(
                    "Safety monitoring error: %s",
                    error,
                )

                self._record_event(
                    "monitor_error",
                    error=str(error),
                )

            self._stop_event.wait(
                self.monitor_interval
            )

    # ============================================================
    # MAIN SAFETY CHECK
    # ============================================================

    def perform_safety_check(
        self,
    ) -> Dict[str, Any]:
        """
        Perform complete robot safety check.
        """

        results = {
            "safe": True,
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "warnings": [],
            "critical": [],
        }

        if not self.enabled:

            results["safe"] = False

            results["warnings"].append(
                "Safety system disabled."
            )

            return results

        # Obstacle check
        obstacle_result = (
            self.check_obstacles()
        )

        results["checks"]["obstacles"] = (
            obstacle_result
        )

        if not obstacle_result["safe"]:

            results["safe"] = False

            results["critical"].append(
                obstacle_result.get(
                    "message",
                    "Obstacle danger detected.",
                )
            )

        elif obstacle_result.get(
            "warning"
        ):

            results["warnings"].append(
                obstacle_result.get(
                    "message",
                    "Obstacle nearby.",
                )
            )

        # Battery check
        battery_result = (
            self.check_battery()
        )

        results["checks"]["battery"] = (
            battery_result
        )

        if battery_result.get(
            "critical"
        ):

            results["safe"] = False

            results["critical"].append(
                battery_result.get(
                    "message",
                    "Critical battery level.",
                )
            )

        elif battery_result.get(
            "warning"
        ):

            results["warnings"].append(
                battery_result.get(
                    "message",
                    "Low battery.",
                )
            )

        # Temperature check
        temperature_result = (
            self.check_temperature()
        )

        results["checks"]["temperature"] = (
            temperature_result
        )

        if not temperature_result["safe"]:

            results["safe"] = False

            results["critical"].append(
                temperature_result.get(
                    "message",
                    "Temperature too high.",
                )
            )

        # Sensor health
        sensor_result = (
            self.check_sensors()
        )

        results["checks"]["sensors"] = (
            sensor_result
        )

        if sensor_result.get(
            "critical"
        ):

            results["safe"] = False

            results["critical"].append(
                sensor_result.get(
                    "message",
                    "Critical sensor failure.",
                )
            )

        # Emergency action
        if (
            not results["safe"]
            and results["critical"]
        ):

            self.emergency_stop(
                reason="; ".join(
                    results["critical"]
                )
            )

        return results

    # ============================================================
    # OBSTACLE SAFETY
    # ============================================================

    def check_obstacles(
        self,
    ) -> Dict[str, Any]:
        """
        Check distance sensors for obstacles.
        """

        result = {
            "safe": True,
            "warning": False,
            "distance": None,
            "message": "",
        }

        distance = (
            self._get_obstacle_distance()
        )

        if distance is None:

            result["message"] = (
                "Obstacle distance unavailable."
            )

            return result

        result["distance"] = distance

        if (
            distance
            <= self.min_obstacle_distance
        ):

            result["safe"] = False

            result["message"] = (
                f"Critical obstacle detected at "
                f"{distance:.2f} cm."
            )

            self._record_event(
                "critical_obstacle",
                distance=distance,
            )

        elif (
            distance
            <= self.warning_obstacle_distance
        ):

            result["warning"] = True

            result["message"] = (
                f"Obstacle nearby at "
                f"{distance:.2f} cm."
            )

            self._record_event(
                "obstacle_warning",
                distance=distance,
            )

        return result

    def _get_obstacle_distance(
        self,
    ) -> Optional[float]:
        """
        Attempt to retrieve obstacle distance
        from SensorManager.
        """

        if self.sensor_manager is None:
            return None

        try:

            if hasattr(
                self.sensor_manager,
                "get_sensor",
            ):

                sensor = (
                    self.sensor_manager
                    .get_sensor(
                        "distance"
                    )
                )

                if sensor:

                    value = sensor.get(
                        "value"
                    )

                    if value is not None:

                        return float(
                            value
                        )

        except Exception as error:

            logger.debug(
                "Distance sensor error: %s",
                error,
            )

        return None

    # ============================================================
    # BATTERY SAFETY
    # ============================================================

    def check_battery(
        self,
    ) -> Dict[str, Any]:
        """
        Check battery level.
        """

        result = {
            "safe": True,
            "warning": False,
            "critical": False,
            "battery_level": None,
            "message": "",
        }

        battery = (
            self._get_battery_level()
        )

        if battery is None:

            result["message"] = (
                "Battery information unavailable."
            )

            return result

        result["battery_level"] = battery

        if (
            battery
            <= self.critical_battery_level
        ):

            result["safe"] = False
            result["critical"] = True

            result["message"] = (
                f"Critical battery level: "
                f"{battery:.1f}%."
            )

            self._record_event(
                "critical_battery",
                battery=battery,
            )

        elif (
            battery
            <= self.low_battery_level
        ):

            result["warning"] = True

            result["message"] = (
                f"Low battery: "
                f"{battery:.1f}%."
            )

            self._record_event(
                "low_battery",
                battery=battery,
            )

        return result

    def _get_battery_level(
        self,
    ) -> Optional[float]:
        """
        Retrieve battery level.
        """

        if self.sensor_manager is None:
            return None

        try:

            if hasattr(
                self.sensor_manager,
                "get_sensor",
            ):

                sensor = (
                    self.sensor_manager
                    .get_sensor(
                        "battery"
                    )
                )

                if sensor:

                    value = sensor.get(
                        "value"
                    )

                    if value is not None:

                        return float(
                            value
                        )

        except Exception as error:

            logger.debug(
                "Battery sensor error: %s",
                error,
            )

        return None

    # ============================================================
    # TEMPERATURE SAFETY
    # ============================================================

    def check_temperature(
        self,
    ) -> Dict[str, Any]:
        """
        Check robot and motor temperature.
        """

        result = {
            "safe": True,
            "temperature": None,
            "message": "",
        }

        temperature = (
            self._get_temperature()
        )

        if temperature is None:

            result["message"] = (
                "Temperature information unavailable."
            )

            return result

        result["temperature"] = temperature

        if (
            temperature
            >= self.max_temperature
        ):

            result["safe"] = False

            result["message"] = (
                f"Critical temperature: "
                f"{temperature:.1f}°C."
            )

            self._record_event(
                "critical_temperature",
                temperature=temperature,
            )

        return result

    def _get_temperature(
        self,
    ) -> Optional[float]:
        """
        Retrieve temperature from sensors.
        """

        if self.sensor_manager is None:
            return None

        try:

            if hasattr(
                self.sensor_manager,
                "get_sensor",
            ):

                sensor = (
                    self.sensor_manager
                    .get_sensor(
                        "temperature"
                    )
                )

                if sensor:

                    value = sensor.get(
                        "value"
                    )

                    if value is not None:

                        return float(
                            value
                        )

        except Exception as error:

            logger.debug(
                "Temperature sensor error: %s",
                error,
            )

        return None

    # ============================================================
    # SENSOR SAFETY
    # ============================================================

    def check_sensors(
        self,
    ) -> Dict[str, Any]:
        """
        Check sensor health.
        """

        result = {
            "safe": True,
            "critical": False,
            "message": "",
            "failed_sensors": [],
        }

        if self.sensor_manager is None:

            result["message"] = (
                "Sensor manager unavailable."
            )

            return result

        try:

            if hasattr(
                self.sensor_manager,
                "get_status",
            ):

                status = (
                    self.sensor_manager
                    .get_status()
                )

                sensors = status.get(
                    "sensors",
                    {}
                )

                for name, sensor in (
                    sensors.items()
                ):

                    if not sensor.get(
                        "active",
                        True,
                    ):

                        result[
                            "failed_sensors"
                        ].append(
                            name
                        )

                if result[
                    "failed_sensors"
                ]:

                    result["safe"] = False

                    result["critical"] = True

                    result["message"] = (
                        "Sensor failure detected: "
                        + ", ".join(
                            result[
                                "failed_sensors"
                            ]
                        )
                    )

        except Exception as error:

            logger.error(
                "Sensor health check error: %s",
                error,
            )

        return result

    # ============================================================
    # MOVEMENT SAFETY
    # ============================================================

    def can_move(
        self,
        speed: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Check whether robot movement is safe.
        """

        result = {
            "allowed": True,
            "reason": "",
        }

        if self.emergency_active:

            result["allowed"] = False

            result["reason"] = (
                "Emergency stop is active."
            )

            return result

        if not self.enabled:

            result["allowed"] = False

            result["reason"] = (
                "Safety system is disabled."
            )

            return result

        if (
            speed is not None
            and abs(float(speed))
            > self.max_speed
        ):

            result["allowed"] = False

            result["reason"] = (
                f"Speed exceeds safety limit "
                f"({self.max_speed})."
            )

            return result

        obstacle = (
            self.check_obstacles()
        )

        if not obstacle["safe"]:

            result["allowed"] = False

            result["reason"] = (
                obstacle["message"]
            )

            return result

        battery = (
            self.check_battery()
        )

        if battery["critical"]:

            result["allowed"] = False

            result["reason"] = (
                battery["message"]
            )

            return result

        temperature = (
            self.check_temperature()
        )

        if not temperature["safe"]:

            result["allowed"] = False

            result["reason"] = (
                temperature["message"]
            )

            return result

        return result

    # ============================================================
    # SAFETY ZONES
    # ============================================================

    def add_safety_zone(
        self,
        name: str,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        restricted: bool = True,
    ) -> bool:
        """
        Add a rectangular safety zone.

        Restricted zones cannot be entered
        when restricted=True.
        """

        zone = {
            "name": name,
            "x_min": float(x_min),
            "x_max": float(x_max),
            "y_min": float(y_min),
            "y_max": float(y_max),
            "restricted": restricted,
        }

        with self._lock:

            self.safety_zones.append(
                zone
            )

        self._emit_event(
            "safety_zone_added",
            zone=zone,
        )

        return True

    def remove_safety_zone(
        self,
        name: str,
    ) -> bool:
        """
        Remove safety zone.
        """

        with self._lock:

            for zone in self.safety_zones:

                if (
                    zone["name"]
                    == name
                ):

                    self.safety_zones.remove(
                        zone
                    )

                    self._emit_event(
                        "safety_zone_removed",
                        name=name,
                    )

                    return True

        return False

    def is_position_safe(
        self,
        x: float,
        y: float,
    ) -> Dict[str, Any]:
        """
        Check whether a position is safe.
        """

        for zone in self.safety_zones:

            inside_x = (
                zone["x_min"]
                <= x
                <= zone["x_max"]
            )

            inside_y = (
                zone["y_min"]
                <= y
                <= zone["y_max"]
            )

            if (
                inside_x
                and inside_y
                and zone["restricted"]
            ):

                return {
                    "safe": False,
                    "zone": zone["name"],
                    "reason": (
                        "Restricted safety zone."
                    ),
                }

        return {
            "safe": True,
            "zone": None,
            "reason": "",
        }

    # ============================================================
    # EMERGENCY STOP
    # ============================================================

    def emergency_stop(
        self,
        reason: str = "Emergency stop activated.",
    ) -> None:
        """
        Immediately stop robot movement
        and all connected systems.
        """

        with self._lock:

            if self.emergency_active:

                return

            self.emergency_active = True

            self.emergency_reason = reason

            self.last_emergency_time = (
                datetime.now().isoformat()
            )

        logger.critical(
            "ROBOT EMERGENCY STOP: %s",
            reason,
        )

        # Stop navigation
        if self.navigation is not None:

            try:

                if hasattr(
                    self.navigation,
                    "stop_navigation",
                ):

                    self.navigation.stop_navigation()

            except Exception as error:

                logger.error(
                    "Navigation emergency stop error: %s",
                    error,
                )

        # Stop robot
        if self.robot_controller is not None:

            try:

                if hasattr(
                    self.robot_controller,
                    "stop",
                ):

                    self.robot_controller.stop()

                elif hasattr(
                    self.robot_controller,
                    "emergency_stop",
                ):

                    self.robot_controller.emergency_stop()

            except Exception as error:

                logger.error(
                    "Robot emergency stop error: %s",
                    error,
                )

        # Stop servos safely
        if self.servo_controller is not None:

            try:

                if hasattr(
                    self.servo_controller,
                    "emergency_stop",
                ):

                    self.servo_controller.emergency_stop()

                elif hasattr(
                    self.servo_controller,
                    "stop_all",
                ):

                    self.servo_controller.stop_all()

            except Exception as error:

                logger.error(
                    "Servo emergency stop error: %s",
                    error,
                )

        self._record_event(
            "emergency_stop",
            reason=reason,
        )

        self._emit_event(
            "emergency_stop",
            reason=reason,
        )

    def reset_emergency(
        self,
    ) -> bool:
        """
        Reset emergency state.

        A complete safety check is performed
        before allowing the reset.
        """

        if not self.emergency_active:

            return True

        results = (
            self.perform_safety_check()
        )

        if not results["safe"]:

            logger.warning(
                "Emergency cannot be reset. "
                "Unsafe conditions still exist."
            )

            return False

        with self._lock:

            self.emergency_active = False
            self.emergency_reason = None

        self._emit_event(
            "emergency_reset"
        )

        logger.info(
            "Robot emergency reset."
        )

        return True

    # ============================================================
    # EVENT HISTORY
    # ============================================================

    def _record_event(
        self,
        event_type: str,
        **data: Any,
    ) -> None:
        """
        Record safety event.
        """

        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            **data,
        }

        with self._lock:

            self.safety_history.append(
                event
            )

            if (
                len(self.safety_history)
                > self.max_history
            ):

                self.safety_history = (
                    self.safety_history[
                        -self.max_history:
                    ]
                )

    def get_history(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Return safety event history.
        """

        limit = max(
            1,
            int(limit),
        )

        return self.safety_history[
            -limit:
        ]

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
        Register safety event handler.
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
        Remove event handler.
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
        Emit safety event.
        """

        event = {
            "event": event_name,
            "timestamp": datetime.now().isoformat(),
            "emergency_active": (
                self.emergency_active
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
                    "Safety event handler error: %s",
                    error,
                )

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return current safety status.
        """

        return {
            "enabled": self.enabled,
            "monitoring": self.monitoring,
            "emergency_active": (
                self.emergency_active
            ),
            "emergency_reason": (
                self.emergency_reason
            ),
            "last_emergency_time": (
                self.last_emergency_time
            ),
            "min_obstacle_distance": (
                self.min_obstacle_distance
            ),
            "warning_obstacle_distance": (
                self.warning_obstacle_distance
            ),
            "max_temperature": (
                self.max_temperature
            ),
            "low_battery_level": (
                self.low_battery_level
            ),
            "critical_battery_level": (
                self.critical_battery_level
            ),
            "max_speed": (
                self.max_speed
            ),
            "safety_zones": len(
                self.safety_zones
            ),
            "history_size": len(
                self.safety_history
            ),
        }

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Safely shutdown RobotSafety.
        """

        logger.info(
            "Shutting down RobotSafety..."
        )

        self.stop_monitoring()

        with self._lock:

            self.event_handlers.clear()

        logger.info(
            "RobotSafety shutdown complete."
        )


RoboticsSafety = RobotSafety


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


    class DemoRobotController:

        def stop(self):

            print(
                "Robot motors stopped."
            )


    safety = RobotSafety(
        robot_controller=(
            DemoRobotController()
        ),
        config={
            "min_obstacle_distance": 20,
            "warning_obstacle_distance": 40,
            "max_temperature": 80,
            "low_battery_level": 20,
            "critical_battery_level": 10,
        },
    )

    safety.add_safety_zone(
        name="Restricted Area",
        x_min=100,
        x_max=200,
        y_min=100,
        y_max=200,
    )

    print(
        safety.is_position_safe(
            50,
            50,
        )
    )

    print(
        safety.get_status()
    )

    safety.start_monitoring()

    time.sleep(1)

    safety.stop_monitoring()

    safety.shutdown()


