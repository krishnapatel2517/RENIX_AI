"""
RENIX Robot Controller
======================

High-level robot control system for RENIX.

Features:

* Connect and disconnect robots
* Forward and backward movement
* Left and right turning
* Stop and emergency stop
* Speed control
* Direction control
* Command execution
* Custom command handlers
* Controller state tracking
* Command history
* Event callbacks
* Hardware adapter support

This controller is hardware-independent.

A hardware adapter can be supplied for:

* Arduino
* Raspberry Pi
* ESP32
* Serial communication
* Bluetooth
* WiFi
* ROS
* HTTP APIs
* Custom robots

Expected adapter methods can include:

connect()
disconnect()
send(command)
move(direction, speed)
stop()
emergency_stop()
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(
"RENIX.Robotics.RobotController"
)

class RobotController:
    """
    High-level controller for a robot.

    This class translates RENIX commands into
    standardized robot actions.
    """

    VALID_DIRECTIONS = {
        "forward",
        "backward",
        "left",
        "right",
        "stop",
    }

    def __init__(
        self,
        robot_id: str = "default_robot",
        adapter: Optional[Any] = None,
        config: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:
        """
        Initialize RobotController.

        Args:
            robot_id:
                Unique robot identifier.

            adapter:
                Hardware communication adapter.

            config:
                Controller configuration.
        """

        self.robot_id = (
            str(robot_id)
            .strip()
            .lower()
            .replace(
                " ",
                "_",
            )
        )

        self.adapter = adapter

        self.config = config or {}

        self.connected = False

        self.running = False

        self.current_direction = "stop"

        self.current_speed = 0

        self.max_speed = int(
            self.config.get(
                "max_speed",
                100,
            )
        )

        self.default_speed = int(
            self.config.get(
                "default_speed",
                50,
            )
        )

        self.min_speed = int(
            self.config.get(
                "min_speed",
                0,
            )
        )

        self.emergency_stopped = False

        self.last_command: Optional[
            Dict[str, Any]
        ] = None

        self.command_handlers: Dict[
            str,
            Callable[..., Any]
        ] = {}

        self.command_history: List[
            Dict[str, Any]
        ] = []

        self.max_history = int(
            self.config.get(
                "max_history",
                500,
            )
        )

        self.event_handlers: List[
            Callable[
                [Dict[str, Any]],
                None,
            ]
        ] = []

        self._lock = threading.RLock()

        self._register_default_commands()

        logger.info(
            "RobotController initialized for %s.",
            self.robot_id,
        )

    # ============================================================
    # CONNECTION
    # ============================================================

    def connect(
        self,
    ) -> bool:
        """
        Connect to the robot hardware.
        """

        with self._lock:

            if self.connected:

                return True

        success = True

        try:

            if self.adapter is not None:

                if hasattr(
                    self.adapter,
                    "connect",
                ):

                    result = (
                        self.adapter.connect()
                    )

                    if result is False:

                        success = False

        except Exception as error:

            logger.error(
                "Failed to connect robot %s: %s",
                self.robot_id,
                error,
            )

            success = False

        with self._lock:

            self.connected = success

            self.running = success

            if success:

                self.emergency_stopped = False

        if success:

            self._emit_event(
                "robot_connected"
            )

            logger.info(
                "Robot connected: %s",
                self.robot_id,
            )

        return success

    def disconnect(
        self,
    ) -> bool:
        """
        Disconnect robot hardware.
        """

        try:

            self.stop()

        except Exception:

            pass

        success = True

        try:

            if self.adapter is not None:

                if hasattr(
                    self.adapter,
                    "disconnect",
                ):

                    result = (
                        self.adapter.disconnect()
                    )

                    if result is False:

                        success = False

        except Exception as error:

            logger.error(
                "Failed to disconnect robot %s: %s",
                self.robot_id,
                error,
            )

            success = False

        with self._lock:

            self.connected = False

            self.running = False

            self.current_direction = "stop"

            self.current_speed = 0

        self._emit_event(
            "robot_disconnected",
            success=success,
        )

        return success

    # ============================================================
    # START / STOP
    # ============================================================

    def start(
        self,
    ) -> bool:
        """
        Start robot controller.
        """

        return self.connect()

    def stop_controller(
        self,
    ) -> bool:
        """
        Stop controller and disconnect robot.
        """

        return self.disconnect()

    # ============================================================
    # MOVEMENT
    # ============================================================

    def move(
        self,
        direction: str,
        speed: Optional[
            int
        ] = None,
        duration: Optional[
            float
        ] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Move the robot.

        Args:
            direction:
                forward, backward, left or right.

            speed:
                Movement speed.

            duration:
                Optional duration in seconds.
        """

        direction = (
            str(direction)
            .lower()
            .strip()
        )

        if direction == "stop":

            return self.stop()

        if (
            direction
            not in self.VALID_DIRECTIONS
        ):

            return {
                "success": False,
                "error": (
                    f"Invalid direction: "
                    f"{direction}"
                ),
            }

        if self.emergency_stopped:

            return {
                "success": False,
                "error": (
                    "Robot is emergency stopped. "
                    "Reset emergency stop first."
                ),
            }

        if not self.connected:

            return {
                "success": False,
                "error": (
                    "Robot is not connected."
                ),
            }

        if speed is None:

            speed = (
                self.default_speed
            )

        speed = (
            self._normalize_speed(
                speed
            )
        )

        success = True
        result = None
        error_message = None

        try:

            if (
                self.adapter is not None
            ):

                if hasattr(
                    self.adapter,
                    "move",
                ):

                    result = (
                        self.adapter.move(
                            direction=direction,
                            speed=speed,
                            duration=duration,
                            **kwargs,
                        )
                    )

                elif hasattr(
                    self.adapter,
                    "send",
                ):

                    payload = {
                        "action": "move",
                        "direction": direction,
                        "speed": speed,
                        "duration": duration,
                        **kwargs,
                    }

                    result = (
                        self.adapter.send(
                            payload
                        )
                    )

                if result is False:

                    success = False

        except Exception as error:

            logger.error(
                "Movement failed: %s",
                error,
            )

            success = False

            error_message = str(
                error
            )

        if success:

            with self._lock:

                self.current_direction = (
                    direction
                )

                self.current_speed = (
                    speed
                )

        command_result = {
            "success": success,
            "robot_id": (
                self.robot_id
            ),
            "direction": direction,
            "speed": speed,
            "duration": duration,
            "result": result,
            "error": error_message,
        }

        self._record_command(
            {
                "command": "move",
                **command_result,
            }
        )

        self._emit_event(
            "robot_moved",
            **command_result,
        )

        return command_result

    def move_forward(
        self,
        speed: Optional[
            int
        ] = None,
        duration: Optional[
            float
        ] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Move robot forward.
        """

        return self.move(
            "forward",
            speed,
            duration,
            **kwargs,
        )

    def move_backward(
        self,
        speed: Optional[
            int
        ] = None,
        duration: Optional[
            float
        ] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Move robot backward.
        """

        return self.move(
            "backward",
            speed,
            duration,
            **kwargs,
        )

    def turn_left(
        self,
        speed: Optional[
            int
        ] = None,
        duration: Optional[
            float
        ] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Turn robot left.
        """

        return self.move(
            "left",
            speed,
            duration,
            **kwargs,
        )

    def turn_right(
        self,
        speed: Optional[
            int
        ] = None,
        duration: Optional[
            float
        ] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Turn robot right.
        """

        return self.move(
            "right",
            speed,
            duration,
            **kwargs,
        )

    # ============================================================
    # STOP
    # ============================================================

    def stop(
        self,
    ) -> Dict[str, Any]:
        """
        Immediately stop normal movement.
        """

        success = True
        result = None

        try:

            if (
                self.adapter is not None
            ):

                if hasattr(
                    self.adapter,
                    "stop",
                ):

                    result = (
                        self.adapter.stop()
                    )

                elif hasattr(
                    self.adapter,
                    "send",
                ):

                    result = (
                        self.adapter.send(
                            {
                                "action": "stop"
                            }
                        )
                    )

                if result is False:

                    success = False

        except Exception as error:

            logger.error(
                "Robot stop failed: %s",
                error,
            )

            success = False

        with self._lock:

            self.current_direction = (
                "stop"
            )

            self.current_speed = 0

        response = {
            "success": success,
            "robot_id": (
                self.robot_id
            ),
            "action": "stop",
            "result": result,
        }

        self._record_command(
            response
        )

        self._emit_event(
            "robot_stopped",
            **response,
        )

        return response

    # ============================================================
    # EMERGENCY STOP
    # ============================================================

    def emergency_stop(
        self,
        reason: Optional[
            str
        ] = None,
    ) -> Dict[str, Any]:
        """
        Perform emergency stop.

        Emergency stop blocks future movement
        until reset_emergency_stop() is called.
        """

        success = True
        result = None

        try:

            if (
                self.adapter is not None
            ):

                if hasattr(
                    self.adapter,
                    "emergency_stop",
                ):

                    result = (
                        self.adapter
                        .emergency_stop()
                    )

                elif hasattr(
                    self.adapter,
                    "stop",
                ):

                    result = (
                        self.adapter.stop()
                    )

                elif hasattr(
                    self.adapter,
                    "send",
                ):

                    result = (
                        self.adapter.send(
                            {
                                "action": (
                                    "emergency_stop"
                                )
                            }
                        )
                    )

                if result is False:

                    success = False

        except Exception as error:

            logger.exception(
                "Emergency stop failed."
            )

            success = False

            result = str(
                error
            )

        with self._lock:

            self.emergency_stopped = True

            self.current_direction = (
                "stop"
            )

            self.current_speed = 0

        response = {
            "success": success,
            "robot_id": (
                self.robot_id
            ),
            "action": (
                "emergency_stop"
            ),
            "reason": (
                reason
                or "Emergency stop activated."
            ),
            "result": result,
        }

        self._record_command(
            response
        )

        self._emit_event(
            "emergency_stop",
            **response,
        )

        logger.warning(
            "EMERGENCY STOP activated for %s",
            self.robot_id,
        )

        return response

    def reset_emergency_stop(
        self,
    ) -> bool:
        """
        Reset emergency stop state.

        Hardware safety systems may require
        additional physical confirmation.
        """

        try:

            if (
                self.adapter is not None
                and hasattr(
                    self.adapter,
                    "reset_emergency_stop",
                )
            ):

                result = (
                    self.adapter
                    .reset_emergency_stop()
                )

                if result is False:

                    return False

        except Exception as error:

            logger.error(
                "Failed to reset emergency stop: %s",
                error,
            )

            return False

        with self._lock:

            self.emergency_stopped = False

        self._emit_event(
            "emergency_stop_reset"
        )

        return True

    # ============================================================
    # SPEED
    # ============================================================

    def set_speed(
        self,
        speed: int,
    ) -> Dict[str, Any]:
        """
        Set robot movement speed.
        """

        speed = (
            self._normalize_speed(
                speed
            )
        )

        success = True
        result = None

        try:

            if (
                self.adapter is not None
            ):

                if hasattr(
                    self.adapter,
                    "set_speed",
                ):

                    result = (
                        self.adapter.set_speed(
                            speed
                        )
                    )

                elif hasattr(
                    self.adapter,
                    "send",
                ):

                    result = (
                        self.adapter.send(
                            {
                                "action": (
                                    "set_speed"
                                ),
                                "speed": speed,
                            }
                        )
                    )

                if result is False:

                    success = False

        except Exception as error:

            logger.error(
                "Failed to set speed: %s",
                error,
            )

            success = False

        if success:

            with self._lock:

                self.current_speed = (
                    speed
                )

        response = {
            "success": success,
            "speed": speed,
            "result": result,
        }

        self._record_command(
            {
                "command": (
                    "set_speed"
                ),
                **response,
            }
        )

        return response

    def increase_speed(
        self,
        amount: int = 10,
    ) -> Dict[str, Any]:
        """
        Increase current speed.
        """

        new_speed = (
            self.current_speed
            + int(amount)
        )

        return self.set_speed(
            new_speed
        )

    def decrease_speed(
        self,
        amount: int = 10,
    ) -> Dict[str, Any]:
        """
        Decrease current speed.
        """

        new_speed = (
            self.current_speed
            - int(amount)
        )

        return self.set_speed(
            new_speed
        )

    # ============================================================
    # COMMAND EXECUTION
    # ============================================================

    def _register_default_commands(
        self,
    ) -> None:
        """
        Register built-in commands.
        """

        self.command_handlers = {
            "forward": (
                self.move_forward
            ),
            "move_forward": (
                self.move_forward
            ),
            "backward": (
                self.move_backward
            ),
            "move_backward": (
                self.move_backward
            ),
            "left": (
                self.turn_left
            ),
            "turn_left": (
                self.turn_left
            ),
            "right": (
                self.turn_right
            ),
            "turn_right": (
                self.turn_right
            ),
            "stop": self.stop,
            "emergency_stop": (
                self.emergency_stop
            ),
            "set_speed": (
                self.set_speed
            ),
            "increase_speed": (
                self.increase_speed
            ),
            "decrease_speed": (
                self.decrease_speed
            ),
        }

    def register_command(
        self,
        command: str,
        handler: Callable[
            ...,
            Any
        ],
    ) -> bool:
        """
        Register a custom command handler.
        """

        if not callable(
            handler
        ):

            return False

        command = (
            str(command)
            .lower()
            .strip()
        )

        with self._lock:

            self.command_handlers[
                command
            ] = handler

        return True

    def unregister_command(
        self,
        command: str,
    ) -> bool:
        """
        Remove a command handler.
        """

        command = (
            str(command)
            .lower()
            .strip()
        )

        with self._lock:

            if (
                command
                not in self.command_handlers
            ):

                return False

            del self.command_handlers[
                command
            ]

        return True

    def execute(
        self,
        command: str,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a robot command.
        """

        command = (
            str(command)
            .lower()
            .strip()
        )

        handler = (
            self.command_handlers.get(
                command
            )
        )

        if handler is not None:

            try:

                result = handler(
                    **kwargs
                )

                self.last_command = {
                    "command": command,
                    "kwargs": kwargs,
                    "timestamp": (
                        datetime.now()
                        .isoformat()
                    ),
                }

                return result

            except TypeError:

                try:

                    return handler()

                except Exception as error:

                    return {
                        "success": False,
                        "error": str(
                            error
                        ),
                    }

            except Exception as error:

                logger.exception(
                    "Robot command failed: %s",
                    command,
                )

                return {
                    "success": False,
                    "error": str(
                        error
                    ),
                }

        return self._send_custom_command(
            command,
            **kwargs,
        )

    def send_command(
        self,
        command: str,
        **kwargs: Any,
    ) -> Any:
        """
        Alias for execute().
        """

        return self.execute(
            command,
            **kwargs,
        )

    def handle_command(
        self,
        command: str,
        **kwargs: Any,
    ) -> Any:
        """
        Alias for execute().
        """

        return self.execute(
            command,
            **kwargs,
        )

    # ============================================================
    # CUSTOM COMMAND
    # ============================================================

    def _send_custom_command(
        self,
        command: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Send custom command directly to adapter.
        """

        if not self.connected:

            return {
                "success": False,
                "error": (
                    "Robot is not connected."
                ),
            }

        if self.emergency_stopped:

            return {
                "success": False,
                "error": (
                    "Robot is emergency stopped."
                ),
            }

        if self.adapter is None:

            return {
                "success": False,
                "error": (
                    "No hardware adapter configured."
                ),
            }

        try:

            if hasattr(
                self.adapter,
                "execute",
            ):

                result = (
                    self.adapter.execute(
                        command,
                        **kwargs,
                    )
                )

            elif hasattr(
                self.adapter,
                "send",
            ):

                result = (
                    self.adapter.send(
                        {
                            "command": command,
                            "parameters": kwargs,
                        }
                    )
                )

            else:

                return {
                    "success": False,
                    "error": (
                        "Adapter does not support "
                        "custom commands."
                    ),
                }

            response = {
                "success": (
                    result is not False
                ),
                "command": command,
                "result": result,
            }

        except Exception as error:

            response = {
                "success": False,
                "command": command,
                "error": str(
                    error
                ),
            }

        self._record_command(
            response
        )

        return response

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return controller status.
        """

        adapter_status = None

        try:

            if (
                self.adapter is not None
                and hasattr(
                    self.adapter,
                    "get_status",
                )
            ):

                adapter_status = (
                    self.adapter
                    .get_status()
                )

        except Exception as error:

            adapter_status = {
                "error": str(
                    error
                )
            }

        return {
            "robot_id": self.robot_id,
            "connected": self.connected,
            "running": self.running,
            "direction": (
                self.current_direction
            ),
            "speed": (
                self.current_speed
            ),
            "max_speed": (
                self.max_speed
            ),
            "emergency_stopped": (
                self.emergency_stopped
            ),
            "last_command": (
                self.last_command
            ),
            "adapter_status": (
                adapter_status
            ),
        }

    def health_check(
        self,
    ) -> Dict[str, Any]:
        """
        Perform controller health check.
        """

        healthy = (
            self.connected
            and not self.emergency_stopped
        )

        adapter_health = None

        try:

            if (
                self.adapter is not None
                and hasattr(
                    self.adapter,
                    "health_check",
                )
            ):

                adapter_health = (
                    self.adapter
                    .health_check()
                )

                if isinstance(
                    adapter_health,
                    bool,
                ):

                    healthy = (
                        healthy
                        and adapter_health
                    )

                elif isinstance(
                    adapter_health,
                    dict,
                ):

                    healthy = (
                        healthy
                        and adapter_health.get(
                            "healthy",
                            True,
                        )
                    )

        except Exception as error:

            healthy = False

            adapter_health = {
                "error": str(
                    error
                )
            }

        return {
            "healthy": healthy,
            "robot_id": self.robot_id,
            "adapter_health": (
                adapter_health
            ),
        }

    # ============================================================
    # HISTORY
    # ============================================================

    def _record_command(
        self,
        command: Dict[
            str,
            Any,
        ],
    ) -> None:
        """
        Record command in history.
        """

        record = dict(
            command
        )

        record.setdefault(
            "timestamp",
            datetime.now().isoformat(),
        )

        with self._lock:

            self.command_history.append(
                record
            )

            if (
                len(
                    self.command_history
                )
                > self.max_history
            ):

                overflow = (
                    len(
                        self.command_history
                    )
                    - self.max_history
                )

                self.command_history = (
                    self.command_history[
                        overflow:
                    ]
                )

    def get_command_history(
        self,
        limit: int = 100,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Return command history.
        """

        with self._lock:

            return [
                dict(record)
                for record in (
                    self.command_history[
                        -max(
                            1,
                            limit,
                        ):
                    ]
                )
            ]

    def clear_command_history(
        self,
    ) -> None:
        """
        Clear command history.
        """

        with self._lock:

            self.command_history.clear()

    # ============================================================
    # EVENT SYSTEM
    # ============================================================

    def add_event_handler(
        self,
        handler: Callable[
            [Dict[str, Any]],
            None,
        ],
    ) -> bool:
        """
        Register event handler.
        """

        if not callable(
            handler
        ):

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
        Emit controller event.
        """

        event = {
            "event": event_name,
            "robot_id": (
                self.robot_id
            ),
            "timestamp": (
                datetime.now().isoformat()
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
                    "Robot event handler failed: %s",
                    error,
                )

    # ============================================================
    # HELPERS
    # ============================================================

    def _normalize_speed(
        self,
        speed: int,
    ) -> int:
        """
        Normalize speed inside allowed range.
        """

        try:

            speed = int(
                speed
            )

        except (
            TypeError,
            ValueError,
        ):

            speed = self.default_speed

        return max(
            self.min_speed,
            min(
                speed,
                self.max_speed,
            ),
        )

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Shutdown RobotController safely.
        """

        logger.info(
            "Shutting down RobotController..."
        )

        try:

            self.stop()

        except Exception:

            pass

        try:

            self.disconnect()

        except Exception:

            pass

        with self._lock:

            self.event_handlers.clear()

        logger.info(
            "RobotController shutdown complete."
        )
