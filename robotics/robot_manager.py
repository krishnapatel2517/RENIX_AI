"""
RENIX Robot Manager
===================

Central management system for robots connected to RENIX.

Features:

* Register multiple robots
* Connect and disconnect robots
* Track robot status
* Manage robot metadata
* Send commands through registered controllers
* Default robot selection
* Robot capability management
* Event callbacks
* Robot health monitoring
* Thread-safe operations

This manager is hardware-agnostic. Actual communication with
Arduino, Raspberry Pi, ESP32, ROS, Bluetooth, Serial, HTTP, etc.
can be implemented through robot adapters/controllers.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("RENIX.Robotics.RobotManager")

class RobotManager:
    """
    Central manager for all robots connected to RENIX.

    Each robot is represented internally as a dictionary containing:

    - id
    - name
    - type
    - controller
    - capabilities
    - metadata
    - connected
    - registered_at
    - last_seen
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize RobotManager.
        """

        self.config = config or {}

        self.robots: Dict[
            str,
            Dict[str, Any]
        ] = {}

        self.default_robot_id: Optional[
            str
        ] = None

        self.event_handlers: List[
            Callable[
                [Dict[str, Any]],
                None,
            ]
        ] = []

        self.command_history: List[
            Dict[str, Any]
        ] = []

        self.max_history = int(
            self.config.get(
                "max_history",
                500,
            )
        )

        self._lock = threading.RLock()

        logger.info(
            "RobotManager initialized."
        )

    # ============================================================
    # ROBOT REGISTRATION
    # ============================================================

    def register_robot(
        self,
        robot_id: str,
        name: Optional[str] = None,
        robot_type: str = "generic",
        controller: Optional[Any] = None,
        capabilities: Optional[
            List[str]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        overwrite: bool = False,
    ) -> bool:
        """
        Register a robot with RENIX.

        Args:
            robot_id:
                Unique identifier.

            name:
                Human-readable robot name.

            robot_type:
                Robot category.

            controller:
                Object responsible for communicating
                with the physical robot.

            capabilities:
                Supported robot capabilities.

            metadata:
                Additional robot information.

            overwrite:
                Allow replacing an existing robot.
        """

        robot_id = self._normalize_robot_id(
            robot_id
        )

        with self._lock:

            if (
                robot_id in self.robots
                and not overwrite
            ):

                logger.warning(
                    "Robot already registered: %s",
                    robot_id,
                )

                return False

            now = datetime.now()

            robot = {
                "id": robot_id,
                "name": (
                    name
                    or robot_id
                ),
                "type": robot_type,
                "controller": controller,
                "capabilities": set(
                    capabilities or []
                ),
                "metadata": dict(
                    metadata or {}
                ),
                "connected": False,
                "registered_at": (
                    now.isoformat()
                ),
                "last_seen": None,
                "status": "offline",
            }

            self.robots[
                robot_id
            ] = robot

            if (
                self.default_robot_id
                is None
            ):

                self.default_robot_id = (
                    robot_id
                )

            event = {
                "event": (
                    "robot_registered"
                ),
                "robot_id": robot_id,
                "name": (
                    robot["name"]
                ),
                "robot_type": (
                    robot_type
                ),
                "timestamp": (
                    now.isoformat()
                ),
            }

        logger.info(
            "Robot registered: %s",
            robot_id,
        )

        self._notify(
            event
        )

        return True

    # ============================================================
    # UNREGISTER ROBOT
    # ============================================================

    def unregister_robot(
        self,
        robot_id: str,
    ) -> bool:
        """
        Remove a robot from RENIX.
        """

        robot_id = self._normalize_robot_id(
            robot_id
        )

        with self._lock:

            robot = (
                self.robots.get(
                    robot_id
                )
            )

            if robot is None:

                return False

            if robot.get(
                "connected"
            ):

                self.disconnect_robot(
                    robot_id
                )

            del self.robots[
                robot_id
            ]

            if (
                self.default_robot_id
                == robot_id
            ):

                self.default_robot_id = (
                    next(
                        iter(
                            self.robots
                        ),
                        None,
                    )
                )

            event = {
                "event": (
                    "robot_unregistered"
                ),
                "robot_id": robot_id,
                "timestamp": (
                    datetime.now().isoformat()
                ),
            }

        logger.info(
            "Robot unregistered: %s",
            robot_id,
        )

        self._notify(
            event
        )

        return True

    # ============================================================
    # CONNECT ROBOT
    # ============================================================

    def connect_robot(
        self,
        robot_id: str,
    ) -> bool:
        """
        Connect to a registered robot.

        The controller may expose either:

        - connect()
        - start()

        Both synchronous and simple controllers
        are supported.
        """

        robot_id = self._normalize_robot_id(
            robot_id
        )

        with self._lock:

            robot = (
                self.robots.get(
                    robot_id
                )
            )

            if robot is None:

                logger.warning(
                    "Robot not found: %s",
                    robot_id,
                )

                return False

            if robot.get(
                "connected"
            ):

                return True

            controller = (
                robot.get(
                    "controller"
                )
            )

        success = True

        try:

            if controller is not None:

                if hasattr(
                    controller,
                    "connect",
                ):

                    result = (
                        controller.connect()
                    )

                    if result is False:

                        success = False

                elif hasattr(
                    controller,
                    "start",
                ):

                    result = (
                        controller.start()
                    )

                    if result is False:

                        success = False

        except Exception as error:

            logger.error(
                "Failed to connect robot %s: %s",
                robot_id,
                error,
            )

            success = False

        with self._lock:

            if not success:

                robot[
                    "connected"
                ] = False

                robot[
                    "status"
                ] = "offline"

                return False

            robot[
                "connected"
            ] = True

            robot[
                "status"
            ] = "online"

            robot[
                "last_seen"
            ] = (
                datetime.now().isoformat()
            )

            event = {
                "event": (
                    "robot_connected"
                ),
                "robot_id": robot_id,
                "timestamp": (
                    datetime.now().isoformat()
                ),
            }

        logger.info(
            "Robot connected: %s",
            robot_id,
        )

        self._notify(
            event
        )

        return True

    # ============================================================
    # DISCONNECT ROBOT
    # ============================================================

    def disconnect_robot(
        self,
        robot_id: str,
    ) -> bool:
        """
        Disconnect from a robot.
        """

        robot_id = self._normalize_robot_id(
            robot_id
        )

        with self._lock:

            robot = (
                self.robots.get(
                    robot_id
                )
            )

            if robot is None:

                return False

            controller = (
                robot.get(
                    "controller"
                )
            )

        success = True

        try:

            if controller is not None:

                if hasattr(
                    controller,
                    "disconnect",
                ):

                    result = (
                        controller.disconnect()
                    )

                    if result is False:

                        success = False

                elif hasattr(
                    controller,
                    "stop",
                ):

                    result = (
                        controller.stop()
                    )

                    if result is False:

                        success = False

        except Exception as error:

            logger.error(
                "Failed to disconnect robot %s: %s",
                robot_id,
                error,
            )

            success = False

        with self._lock:

            robot[
                "connected"
            ] = False

            robot[
                "status"
            ] = "offline"

            event = {
                "event": (
                    "robot_disconnected"
                ),
                "robot_id": robot_id,
                "success": success,
                "timestamp": (
                    datetime.now().isoformat()
                ),
            }

        logger.info(
            "Robot disconnected: %s",
            robot_id,
        )

        self._notify(
            event
        )

        return success

    # ============================================================
    # DEFAULT ROBOT
    # ============================================================

    def set_default_robot(
        self,
        robot_id: str,
    ) -> bool:
        """
        Set the default robot.
        """

        robot_id = self._normalize_robot_id(
            robot_id
        )

        with self._lock:

            if (
                robot_id
                not in self.robots
            ):

                return False

            self.default_robot_id = (
                robot_id
            )

            return True

    def get_default_robot(
        self,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Return default robot information.
        """

        with self._lock:

            if (
                self.default_robot_id
                is None
            ):

                return None

            robot = (
                self.robots.get(
                    self.default_robot_id
                )
            )

            if robot is None:

                return None

            return self._sanitize_robot(
                robot
            )

    # ============================================================
    # ROBOT LOOKUP
    # ============================================================

    def get_robot(
        self,
        robot_id: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Return robot information.
        """

        robot_id = self._normalize_robot_id(
            robot_id
        )

        with self._lock:

            robot = (
                self.robots.get(
                    robot_id
                )
            )

            if robot is None:

                return None

            return self._sanitize_robot(
                robot
            )

    def get_robot_controller(
        self,
        robot_id: Optional[
            str
        ] = None,
    ) -> Optional[Any]:
        """
        Return the controller for a robot.
        """

        with self._lock:

            if robot_id is None:

                robot_id = (
                    self.default_robot_id
                )

            if robot_id is None:

                return None

            robot_id = (
                self._normalize_robot_id(
                    robot_id
                )
            )

            robot = (
                self.robots.get(
                    robot_id
                )
            )

            if robot is None:

                return None

            return robot.get(
                "controller"
            )

    def list_robots(
        self,
        connected_only: bool = False,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        List registered robots.
        """

        with self._lock:

            robots = []

            for robot in (
                self.robots.values()
            ):

                if (
                    connected_only
                    and not robot.get(
                        "connected"
                    )
                ):

                    continue

                robots.append(
                    self._sanitize_robot(
                        robot
                    )
                )

            return robots

    # ============================================================
    # ROBOT CAPABILITIES
    # ============================================================

    def add_capability(
        self,
        robot_id: str,
        capability: str,
    ) -> bool:
        """
        Add a capability to a robot.
        """

        robot_id = self._normalize_robot_id(
            robot_id
        )

        capability = (
            str(capability)
            .lower()
            .strip()
        )

        with self._lock:

            robot = (
                self.robots.get(
                    robot_id
                )
            )

            if robot is None:

                return False

            robot[
                "capabilities"
            ].add(
                capability
            )

            return True

    def remove_capability(
        self,
        robot_id: str,
        capability: str,
    ) -> bool:
        """
        Remove a robot capability.
        """

        robot_id = self._normalize_robot_id(
            robot_id
        )

        capability = (
            str(capability)
            .lower()
            .strip()
        )

        with self._lock:

            robot = (
                self.robots.get(
                    robot_id
                )
            )

            if robot is None:

                return False

            capabilities = (
                robot[
                    "capabilities"
                ]
            )

            if (
                capability
                not in capabilities
            ):

                return False

            capabilities.remove(
                capability
            )

            return True

    def has_capability(
        self,
        robot_id: str,
        capability: str,
    ) -> bool:
        """
        Check whether a robot supports
        a capability.
        """

        robot_id = self._normalize_robot_id(
            robot_id
        )

        capability = (
            str(capability)
            .lower()
            .strip()
        )

        with self._lock:

            robot = (
                self.robots.get(
                    robot_id
                )
            )

            if robot is None:

                return False

            return (
                capability
                in robot[
                    "capabilities"
                ]
            )

    # ============================================================
    # SEND COMMAND
    # ============================================================

    def send_command(
        self,
        command: str,
        robot_id: Optional[
            str
        ] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Send a command to a robot.

        The robot controller can implement one of:

        - execute(command, **kwargs)
        - send_command(command, **kwargs)
        - handle_command(command, **kwargs)
        """

        command = (
            str(command)
            .lower()
            .strip()
        )

        with self._lock:

            if robot_id is None:

                robot_id = (
                    self.default_robot_id
                )

            if robot_id is None:

                return {
                    "success": False,
                    "error": (
                        "No robot specified and "
                        "no default robot configured."
                    ),
                }

            robot_id = (
                self._normalize_robot_id(
                    robot_id
                )
            )

            robot = (
                self.robots.get(
                    robot_id
                )
            )

            if robot is None:

                return {
                    "success": False,
                    "error": (
                        f"Robot '{robot_id}' "
                        "not found."
                    ),
                }

            if not robot.get(
                "connected"
            ):

                return {
                    "success": False,
                    "error": (
                        f"Robot '{robot_id}' "
                        "is not connected."
                    ),
                }

            controller = (
                robot.get(
                    "controller"
                )
            )

        if controller is None:

            return {
                "success": False,
                "error": (
                    "Robot has no controller."
                ),
            }

        result: Any = None
        success = True
        error_message = None

        try:

            if hasattr(
                controller,
                "execute",
            ):

                result = (
                    controller.execute(
                        command,
                        **kwargs,
                    )
                )

            elif hasattr(
                controller,
                "send_command",
            ):

                result = (
                    controller.send_command(
                        command,
                        **kwargs,
                    )
                )

            elif hasattr(
                controller,
                "handle_command",
            ):

                result = (
                    controller.handle_command(
                        command,
                        **kwargs,
                    )
                )

            else:

                success = False

                error_message = (
                    "Controller does not support "
                    "command execution."
                )

        except Exception as error:

            logger.exception(
                "Robot command failed."
            )

            success = False

            error_message = str(
                error
            )

        command_record = {
            "robot_id": robot_id,
            "command": command,
            "kwargs": kwargs,
            "success": success,
            "timestamp": (
                datetime.now().isoformat()
            ),
        }

        if error_message:

            command_record[
                "error"
            ] = error_message

        with self._lock:

            self._add_command_history(
                command_record
            )

            if success:

                robot = (
                    self.robots.get(
                        robot_id
                    )
                )

                if robot is not None:

                    robot[
                        "last_seen"
                    ] = (
                        datetime.now()
                        .isoformat()
                    )

        event = {
            "event": (
                "robot_command_executed"
            ),
            **command_record,
        }

        self._notify(
            event
        )

        return {
            "success": success,
            "robot_id": robot_id,
            "command": command,
            "result": result,
            "error": error_message,
        }

    # ============================================================
    # ROBOT STATUS
    # ============================================================

    def update_robot_status(
        self,
        robot_id: str,
        status: str,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> bool:
        """
        Update robot status.
        """

        robot_id = self._normalize_robot_id(
            robot_id
        )

        status = (
            str(status)
            .lower()
            .strip()
        )

        with self._lock:

            robot = (
                self.robots.get(
                    robot_id
                )
            )

            if robot is None:

                return False

            robot[
                "status"
            ] = status

            robot[
                "last_seen"
            ] = (
                datetime.now().isoformat()
            )

            if metadata:

                robot[
                    "metadata"
                ].update(
                    metadata
                )

            return True

    def refresh_robot_status(
        self,
        robot_id: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Request latest status from robot controller.
        """

        robot_id = self._normalize_robot_id(
            robot_id
        )

        controller = (
            self.get_robot_controller(
                robot_id
            )
        )

        if controller is None:

            return None

        try:

            if hasattr(
                controller,
                "get_status",
            ):

                status = (
                    controller.get_status()
                )

                if isinstance(
                    status,
                    dict,
                ):

                    self.update_robot_status(
                        robot_id,
                        status.get(
                            "status",
                            "online",
                        ),
                        metadata=status,
                    )

                return status

        except Exception as error:

            logger.error(
                "Failed to refresh robot status "
                "%s: %s",
                robot_id,
                error,
            )

        return None

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def health_check(
        self,
        robot_id: Optional[
            str
        ] = None,
    ) -> Dict[str, Any]:
        """
        Perform a basic health check.

        If robot_id is None, checks all robots.
        """

        if robot_id is not None:

            robot_ids = [
                self._normalize_robot_id(
                    robot_id
                )
            ]

        else:

            with self._lock:

                robot_ids = list(
                    self.robots.keys()
                )

        results = {}

        for current_robot_id in robot_ids:

            robot = (
                self.get_robot(
                    current_robot_id
                )
            )

            if robot is None:

                results[
                    current_robot_id
                ] = {
                    "healthy": False,
                    "error": (
                        "Robot not found."
                    ),
                }

                continue

            controller = (
                self.get_robot_controller(
                    current_robot_id
                )
            )

            healthy = robot.get(
                "connected",
                False,
            )

            details = None

            if (
                controller is not None
                and hasattr(
                    controller,
                    "health_check",
                )
            ):

                try:

                    details = (
                        controller.health_check()
                    )

                    if isinstance(
                        details,
                        bool,
                    ):

                        healthy = details

                    elif isinstance(
                        details,
                        dict,
                    ):

                        healthy = bool(
                            details.get(
                                "healthy",
                                healthy,
                            )
                        )

                except Exception as error:

                    healthy = False

                    details = {
                        "error": str(
                            error
                        )
                    }

            results[
                current_robot_id
            ] = {
                "healthy": healthy,
                "details": details,
            }

        return results

    # ============================================================
    # COMMAND HISTORY
    # ============================================================

    def _add_command_history(
        self,
        command_record: Dict[
            str,
            Any,
        ],
    ) -> None:
        """
        Store command history.
        """

        self.command_history.append(
            command_record
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
        Return recent robot commands.
        """

        with self._lock:

            return [
                record.copy()
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
    ) -> None:
        """
        Register an event handler.
        """

        if not callable(
            handler
        ):

            return

        with self._lock:

            if (
                handler
                not in self.event_handlers
            ):

                self.event_handlers.append(
                    handler
                )

    def remove_event_handler(
        self,
        handler: Callable,
    ) -> bool:
        """
        Remove an event handler.
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

    def _notify(
        self,
        event: Dict[str, Any],
    ) -> None:
        """
        Notify registered handlers.
        """

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

    @staticmethod
    def _normalize_robot_id(
        robot_id: str,
    ) -> str:
        """
        Normalize robot identifier.
        """

        if robot_id is None:

            raise ValueError(
                "Robot ID cannot be None."
            )

        robot_id = (
            str(robot_id)
            .strip()
            .lower()
            .replace(
                " ",
                "_",
            )
        )

        if not robot_id:

            raise ValueError(
                "Robot ID cannot be empty."
            )

        return robot_id

    @staticmethod
    def _sanitize_robot(
        robot: Dict[
            str,
            Any,
        ],
    ) -> Dict[
        str,
        Any,
    ]:
        """
        Return robot data without exposing
        controller object.
        """

        return {
            "id": robot.get(
                "id"
            ),
            "name": robot.get(
                "name"
            ),
            "type": robot.get(
                "type"
            ),
            "capabilities": sorted(
                list(
                    robot.get(
                        "capabilities",
                        set(),
                    )
                )
            ),
            "metadata": dict(
                robot.get(
                    "metadata",
                    {},
                )
            ),
            "connected": robot.get(
                "connected",
                False,
            ),
            "status": robot.get(
                "status",
                "unknown",
            ),
            "registered_at": robot.get(
                "registered_at"
            ),
            "last_seen": robot.get(
                "last_seen"
            ),
        }

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return RobotManager status.
        """

        with self._lock:

            total = len(
                self.robots
            )

            connected = sum(
                1
                for robot in (
                    self.robots.values()
                )
                if robot.get(
                    "connected"
                )
            )

            return {
                "total_robots": total,
                "connected_robots": (
                    connected
                ),
                "default_robot_id": (
                    self.default_robot_id
                ),
                "command_history": (
                    len(
                        self.command_history
                    )
                ),
            }

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Disconnect all robots and shut down
        RobotManager.
        """

        logger.info(
            "Shutting down RobotManager..."
        )

        with self._lock:

            robot_ids = list(
                self.robots.keys()
            )

        for robot_id in robot_ids:

            try:

                self.disconnect_robot(
                    robot_id
                )

            except Exception as error:

                logger.error(
                    "Failed to disconnect robot "
                    "%s during shutdown: %s",
                    robot_id,
                    error,
                )

        with self._lock:

            self.event_handlers.clear()

        logger.info(
            "RobotManager shutdown complete."
        )
