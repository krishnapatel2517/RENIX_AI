"""
RENIX AI - Robotics Agent

Responsible for robotics-related operations inside RENIX.

Responsibilities:
- Initialize and manage the robotics subsystem
- Discover available robots/controllers
- Connect and disconnect robots
- Read robot status
- Read sensors
- Control servos
- Move robots
- Navigate
- Manipulate supported objects
- Process robot-vision requests
- Provide safety checks before physical actions
- Maintain robot command history
- Emit events to the RENIX event system

The agent is designed to work with the modules inside:

    RENIX/robotics/

including:

    robot_manager.py
    robot_controller.py
    servo_controller.py
    sensor_manager.py
    robot_vision.py
    navigation.py
    object_manipulation.py
    safety.py

The implementation intentionally supports missing optional modules so that
RENIX can start even when robotics hardware/software is not installed.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence


class RoboticsAgent:
    """
    RENIX robotics-management agent.

    This class acts as the AI-facing interface for the robotics subsystem.
    Higher-level RENIX components can send natural-language-resolved actions
    here without needing to know the implementation details of individual
    robot controllers.
    """

    agent_name = "robotics_agent"
    agent_version = "1.0.0"

    agent_description = (
        "Controls and monitors supported robotic systems, sensors, "
        "servos, navigation and object-manipulation capabilities."
    )

    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        priority: Any = None,
        logger: Optional[logging.Logger] = None,
        event_callback: Any = None,
        confirmation_callback: Any = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.agent_id = (
            agent_id
            or "robotics-agent"
        )

        self.name = (
            name
            or "RENIX Robotics Agent"
        )

        self.description = (
            description
            or self.agent_description
        )

        self.priority = priority

        self.logger = (
            logger
            or logging.getLogger(
                __name__
            )
        )

        self.event_callback = (
            event_callback
        )

        self.confirmation_callback = (
            confirmation_callback
        )

        self.context = (
            dict(context)
            if context
            else {}
        )

        self.enabled = True
        self.initialized = False

        # Robotics subsystem modules.
        self.robot_manager = None
        self.robot_controller = None
        self.servo_controller = None
        self.sensor_manager = None
        self.robot_vision = None
        self.navigation = None
        self.object_manipulation = None
        self.safety = None

        self._modules_loaded = False

        # Internal state.
        self._robots: Dict[
            str,
            Dict[str, Any],
        ] = {}

        self._connections: Dict[
            str,
            Dict[str, Any],
        ] = {}

        self._sensor_cache: Dict[
            str,
            Any,
        ] = {}

        self._servo_state: Dict[
            str,
            Any,
        ] = {}

        self._command_history: List[
            Dict[str, Any]
        ] = []

        self._emergency_stopped = False

    # ======================================================================
    # INITIALIZATION
    # ======================================================================

    async def initialize(self) -> bool:
        """
        Initialize the robotics subsystem.
        """

        if self.initialized:
            return True

        self._load_modules()

        # If the robot manager exposes an initialization method, use it.
        if self.robot_manager is not None:

            await self._call_module(
                self.robot_manager,
                (
                    "initialize",
                    "init",
                    "start",
                ),
            )

        self.initialized = True

        await self._emit_event(
            "robotics_agent_initialized",
            {
                "agent_id": self.agent_id,
                "timestamp": self._timestamp(),
            },
        )

        return True

    async def shutdown(self) -> bool:
        """
        Safely shut down the robotics subsystem.
        """

        # Stop active movement before disconnecting.
        try:
            await self.emergency_stop(
                reason="robotics_agent_shutdown"
            )
        except Exception:
            self.logger.exception(
                "Failed to stop robotics subsystem during shutdown."
            )

        robot_ids = list(
            self._connections.keys()
        )

        for robot_id in robot_ids:

            try:

                await self.disconnect_robot(
                    robot_id
                )

            except Exception:

                self.logger.exception(
                    "Failed to disconnect robot %s.",
                    robot_id,
                )

        if self.robot_manager is not None:

            await self._call_module(
                self.robot_manager,
                (
                    "shutdown",
                    "stop",
                    "close",
                ),
            )

        self.initialized = False

        await self._emit_event(
            "robotics_agent_shutdown",
            {
                "agent_id": self.agent_id,
                "timestamp": self._timestamp(),
            },
        )

        return True

    # ======================================================================
    # MODULE LOADING
    # ======================================================================

    def _load_modules(self) -> None:
        """
        Dynamically load robotics modules.

        Missing modules are intentionally tolerated because robotics hardware
        may not be available on every RENIX installation.
        """

        if self._modules_loaded:
            return

        self._modules_loaded = True

        modules = {
            "robot_manager": (
                "robotics.robot_manager",
                "RobotManager",
            ),
            "robot_controller": (
                "robotics.robot_controller",
                "RobotController",
            ),
            "servo_controller": (
                "robotics.servo_controller",
                "ServoController",
            ),
            "sensor_manager": (
                "robotics.sensor_manager",
                "SensorManager",
            ),
            "robot_vision": (
                "robotics.robot_vision",
                "RobotVision",
            ),
            "navigation": (
                "robotics.navigation",
                "Navigation",
            ),
            "object_manipulation": (
                "robotics.object_manipulation",
                "ObjectManipulation",
            ),
            "safety": (
                "robotics.safety",
                "RobotSafety",
            ),
        }

        for attribute, (
            module_name,
            class_name,
        ) in modules.items():

            instance = self._load_class(
                module_name,
                class_name,
            )

            if instance is not None:

                setattr(
                    self,
                    attribute,
                    instance,
                )

    @staticmethod
    def _load_class(
        module_name: str,
        class_name: str,
    ) -> Any:
        """
        Import and instantiate a class from a module.
        """

        try:

            module = __import__(
                module_name,
                fromlist=[
                    class_name
                ],
            )

            cls = getattr(
                module,
                class_name,
            )

            try:
                return cls()

            except TypeError:
                return cls

        except Exception:

            return None

    # ======================================================================
    # MAIN EXECUTION ROUTER
    # ======================================================================

    async def execute(
        self,
        instruction: Optional[str] = None,
        *,
        action: Optional[str] = None,
        parameters: Optional[
            Dict[str, Any]
        ] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute a robotics operation.

        Examples:

            await agent.execute(
                action="discover"
            )

            await agent.execute(
                action="connect",
                parameters={
                    "robot_id": "robot-01"
                }
            )

            await agent.execute(
                action="move",
                parameters={
                    "robot_id": "robot-01",
                    "direction": "forward",
                    "distance": 1.0
                }
            )
        """

        if not self.initialized:
            await self.initialize()

        params = dict(
            parameters
            or {}
        )

        params.update(
            kwargs
        )

        resolved_action = (
            action
            or params.pop(
                "action",
                None,
            )
            or self._resolve_action(
                instruction
                or ""
            )
        )

        resolved_action = (
            str(
                resolved_action
            )
            .strip()
            .lower()
        )

        handlers = {
            "discover": self.discover_robots,
            "scan": self.discover_robots,
            "robots": self.list_robots,
            "list": self.list_robots,
            "connect": self.connect_robot,
            "disconnect": self.disconnect_robot,
            "status": self.robot_status,
            "info": self.robot_info,
            "move": self.move_robot,
            "drive": self.move_robot,
            "stop": self.stop_robot,
            "emergency_stop": self.emergency_stop,
            "estop": self.emergency_stop,
            "servo": self.control_servo,
            "servo_control": self.control_servo,
            "sensor": self.read_sensor,
            "sensors": self.read_sensors,
            "vision": self.process_vision,
            "see": self.process_vision,
            "navigate": self.navigate,
            "navigation": self.navigate,
            "manipulate": self.manipulate_object,
            "grip": self.manipulate_object,
            "safety": self.safety_status,
            "command": self.send_robot_command,
            "control": self.send_robot_command,
        }

        handler = handlers.get(
            resolved_action
        )

        if handler is None:

            return self._failure(
                (
                    "Unknown robotics action: "
                    f"{resolved_action}"
                )
            )

        try:

            result = handler(
                **params
            )

            if asyncio.iscoroutine(
                result
            ):
                result = await result

            return result

        except Exception as exc:

            self.logger.exception(
                "Robotics action failed: %s",
                resolved_action,
            )

            return self._failure(
                str(exc)
            )

    def _resolve_action(
        self,
        instruction: str,
    ) -> str:
        """
        Resolve common natural-language robotics instructions.
        """

        text = (
            instruction
            .strip()
            .lower()
        )

        if any(
            phrase in text
            for phrase in (
                "discover robots",
                "scan robots",
                "find robots",
                "discover robot",
            )
        ):
            return "discover"

        if "connect" in text:
            return "connect"

        if "disconnect" in text:
            return "disconnect"

        if (
            "emergency stop" in text
            or "e-stop" in text
            or "estop" in text
        ):
            return "emergency_stop"

        if (
            text == "stop"
            or "stop robot" in text
            or "stop the robot" in text
        ):
            return "stop"

        if (
            "move" in text
            or "drive" in text
            or "forward" in text
            or "backward" in text
            or "backwards" in text
            or "left" in text
            or "right" in text
        ):
            return "move"

        if "servo" in text:
            return "servo"

        if "sensor" in text:
            return "sensor"

        if (
            "camera" in text
            or "vision" in text
            or "see" in text
            or "look" in text
        ):
            return "vision"

        if "navigate" in text:
            return "navigate"

        if (
            "grab" in text
            or "grip" in text
            or "pick up" in text
            or "manipulate" in text
        ):
            return "manipulate"

        if "safety" in text:
            return "safety"

        if "status" in text:
            return "status"

        return "command"

    # ======================================================================
    # ROBOT DISCOVERY
    # ======================================================================

    async def discover_robots(
        self,
        robot_type: Optional[str] = None,
        interface: Optional[str] = None,
        timeout: float = 5.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Discover robots/controllers available to RENIX.
        """

        discovered: List[
            Dict[str, Any]
        ] = []

        if self.robot_manager is not None:

            result = await self._call_module(
                self.robot_manager,
                (
                    "discover",
                    "scan",
                    "find_robots",
                    "list_robots",
                ),
                robot_type=robot_type,
                interface=interface,
                timeout=timeout,
                **kwargs,
            )

            if result is not None:

                discovered = (
                    self._normalize_robots(
                        result
                    )
                )

        for robot in discovered:

            robot_id = (
                self._robot_id(
                    robot
                )
            )

            self._robots[
                robot_id
            ] = robot

        await self._emit_event(
            "robots_discovered",
            {
                "count": len(
                    discovered
                ),
                "robot_type": robot_type,
                "interface": interface,
            },
        )

        return {
            "success": True,
            "action": "discover",
            "count": len(
                discovered
            ),
            "robots": discovered,
        }

    async def list_robots(
        self,
        connected_only: bool = False,
        robot_type: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        List known robots.
        """

        robots = list(
            self._robots.values()
        )

        if connected_only:

            robots = [
                robot
                for robot in robots
                if self._robot_id(
                    robot
                )
                in self._connections
            ]

        if robot_type:

            normalized_type = (
                str(
                    robot_type
                ).lower()
            )

            robots = [
                robot
                for robot in robots
                if str(
                    robot.get(
                        "type",
                        "",
                    )
                ).lower()
                == normalized_type
            ]

        return {
            "success": True,
            "action": "list",
            "count": len(
                robots
            ),
            "robots": robots,
        }

    # ======================================================================
    # CONNECTION
    # ======================================================================

    async def connect_robot(
        self,
        robot_id: Optional[str] = None,
        interface: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Connect to a robot.
        """

        if not robot_id:

            return self._failure(
                "robot_id is required"
            )

        if self._emergency_stopped:

            return self._failure(
                (
                    "Robotics system is in "
                    "emergency-stop state."
                )
            )

        robot = self._robots.get(
            robot_id,
            {
                "id": robot_id,
                "name": robot_id,
                "interface": interface,
            },
        )

        if self.robot_manager is not None:

            result = await self._call_module(
                self.robot_manager,
                (
                    "connect",
                    "connect_robot",
                ),
                robot=robot,
                robot_id=robot_id,
                interface=interface,
                **kwargs,
            )

            if result is not None:

                self._connections[
                    robot_id
                ] = {
                    "robot_id": robot_id,
                    "connected_at": self._timestamp(),
                    "interface": interface,
                    "result": result,
                }

                await self._emit_event(
                    "robot_connected",
                    {
                        "robot_id": robot_id,
                    },
                )

                return {
                    "success": True,
                    "action": "connect",
                    "robot_id": robot_id,
                    "result": result,
                }

        robot[
            "connected"
        ] = True

        self._robots[
            robot_id
        ] = robot

        self._connections[
            robot_id
        ] = {
            "robot_id": robot_id,
            "connected_at": self._timestamp(),
            "interface": interface,
        }

        await self._emit_event(
            "robot_connected",
            {
                "robot_id": robot_id,
            },
        )

        return {
            "success": True,
            "action": "connect",
            "robot_id": robot_id,
            "connected": True,
            "method": "fallback",
        }

    async def disconnect_robot(
        self,
        robot_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Disconnect from a robot.
        """

        if not robot_id:

            return self._failure(
                "robot_id is required"
            )

        # Stop before disconnecting.
        try:

            await self.stop_robot(
                robot_id=robot_id,
                internal=True,
            )

        except Exception:

            self.logger.debug(
                "Could not stop robot %s before disconnect.",
                robot_id,
            )

        robot = self._robots.get(
            robot_id
        )

        if self.robot_manager is not None:

            result = await self._call_module(
                self.robot_manager,
                (
                    "disconnect",
                    "disconnect_robot",
                ),
                robot=robot,
                robot_id=robot_id,
                **kwargs,
            )

            if result is not None:

                self._connections.pop(
                    robot_id,
                    None,
                )

                return {
                    "success": True,
                    "action": "disconnect",
                    "robot_id": robot_id,
                    "result": result,
                }

        if robot is not None:

            robot[
                "connected"
            ] = False

        self._connections.pop(
            robot_id,
            None,
        )

        await self._emit_event(
            "robot_disconnected",
            {
                "robot_id": robot_id,
            },
        )

        return {
            "success": True,
            "action": "disconnect",
            "robot_id": robot_id,
            "connected": False,
            "method": "fallback",
        }

    # ======================================================================
    # ROBOT STATUS
    # ======================================================================

    async def robot_status(
        self,
        robot_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Get current robot status.
        """

        if not robot_id:

            return self._failure(
                "robot_id is required"
            )

        robot = self._robots.get(
            robot_id
        )

        if self.robot_manager is not None:

            result = await self._call_module(
                self.robot_manager,
                (
                    "status",
                    "get_status",
                    "robot_status",
                ),
                robot_id=robot_id,
                robot=robot,
                **kwargs,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "status",
                    "robot_id": robot_id,
                    "status": result,
                }

        if robot is None:

            return {
                "success": False,
                "action": "status",
                "robot_id": robot_id,
                "error": "Robot is not known.",
            }

        return {
            "success": True,
            "action": "status",
            "robot_id": robot_id,
            "status": {
                **robot,
                "connected": (
                    robot_id
                    in self._connections
                ),
                "emergency_stopped": (
                    self._emergency_stopped
                ),
            },
        }

    async def robot_info(
        self,
        robot_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Return robot information.
        """

        if robot_id:

            robot = self._robots.get(
                robot_id
            )

            if robot is None:

                return {
                    "success": False,
                    "error": "Robot not found.",
                    "robot_id": robot_id,
                }

            return {
                "success": True,
                "action": "info",
                "robot": robot,
                "connection": self._connections.get(
                    robot_id
                ),
            }

        return await self.list_robots(
            **kwargs
        )

    # ======================================================================
    # MOVEMENT
    # ======================================================================

    async def move_robot(
        self,
        robot_id: Optional[str] = None,
        direction: Optional[str] = None,
        distance: Optional[float] = None,
        speed: Optional[float] = None,
        duration: Optional[float] = None,
        rotation: Optional[float] = None,
        require_confirmation: bool = False,
        internal: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Move a robot.

        Supported logical directions include:

            forward
            backward
            left
            right
            stop

        Physical safety is delegated to the robotics safety module when
        available.
        """

        if not robot_id:

            return self._failure(
                "robot_id is required"
            )

        if self._emergency_stopped:

            return self._failure(
                (
                    "Movement blocked because "
                    "the emergency stop is active."
                )
            )

        if not internal and require_confirmation:

            confirmed = await self._request_confirmation(
                {
                    "action": "move",
                    "robot_id": robot_id,
                    "direction": direction,
                    "distance": distance,
                    "speed": speed,
                    "duration": duration,
                    "rotation": rotation,
                }
            )

            if not confirmed:

                return {
                    "success": False,
                    "action": "move",
                    "robot_id": robot_id,
                    "cancelled": True,
                }

        safety_result = await self._check_safety(
            action="move",
            robot_id=robot_id,
            direction=direction,
            distance=distance,
            speed=speed,
            duration=duration,
            rotation=rotation,
        )

        if not safety_result["safe"]:

            return {
                "success": False,
                "action": "move",
                "robot_id": robot_id,
                "blocked_by_safety": True,
                "reason": safety_result[
                    "reason"
                ],
            }

        if self.robot_controller is not None:

            result = await self._call_module(
                self.robot_controller,
                (
                    "move",
                    "drive",
                    "move_robot",
                ),
                robot_id=robot_id,
                direction=direction,
                distance=distance,
                speed=speed,
                duration=duration,
                rotation=rotation,
                **kwargs,
            )

            if result is not None:

                await self._record_command(
                    "move",
                    {
                        "robot_id": robot_id,
                        "direction": direction,
                        "distance": distance,
                        "speed": speed,
                        "duration": duration,
                        "rotation": rotation,
                    },
                )

                return {
                    "success": True,
                    "action": "move",
                    "robot_id": robot_id,
                    "result": result,
                }

        # Fallback state only.
        robot = self._robots.setdefault(
            robot_id,
            {
                "id": robot_id,
                "name": robot_id,
            },
        )

        robot[
            "last_motion"
        ] = {
            "direction": direction,
            "distance": distance,
            "speed": speed,
            "duration": duration,
            "rotation": rotation,
            "timestamp": self._timestamp(),
        }

        await self._record_command(
            "move",
            {
                "robot_id": robot_id,
                "direction": direction,
                "distance": distance,
                "speed": speed,
                "duration": duration,
                "rotation": rotation,
            },
        )

        return {
            "success": True,
            "action": "move",
            "robot_id": robot_id,
            "direction": direction,
            "distance": distance,
            "speed": speed,
            "duration": duration,
            "rotation": rotation,
            "method": "fallback",
        }

    async def stop_robot(
        self,
        robot_id: Optional[str] = None,
        internal: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Immediately request a normal robot stop.
        """

        if not robot_id:

            return self._failure(
                "robot_id is required"
            )

        if self.robot_controller is not None:

            result = await self._call_module(
                self.robot_controller,
                (
                    "stop",
                    "stop_robot",
                    "halt",
                ),
                robot_id=robot_id,
                **kwargs,
            )

            if result is not None:

                await self._record_command(
                    "stop",
                    {
                        "robot_id": robot_id,
                    },
                )

                return {
                    "success": True,
                    "action": "stop",
                    "robot_id": robot_id,
                    "result": result,
                }

        robot = self._robots.get(
            robot_id
        )

        if robot is not None:

            robot[
                "last_motion"
            ] = {
                "direction": "stop",
                "timestamp": self._timestamp(),
            }

        await self._record_command(
            "stop",
            {
                "robot_id": robot_id,
            },
        )

        return {
            "success": True,
            "action": "stop",
            "robot_id": robot_id,
            "stopped": True,
            "method": "fallback",
        }

    async def emergency_stop(
        self,
        robot_id: Optional[str] = None,
        reason: str = "manual_emergency_stop",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Activate the RENIX robotics emergency-stop state.

        If a robot ID is supplied, only that robot is targeted when the
        underlying safety subsystem supports targeted stopping.

        If no robot ID is supplied, the global RENIX robotics emergency-stop
        state is activated.
        """

        self._emergency_stopped = True

        result = None

        if self.safety is not None:

            result = await self._call_module(
                self.safety,
                (
                    "emergency_stop",
                    "emergency_stop_all",
                    "stop_all",
                ),
                robot_id=robot_id,
                reason=reason,
                **kwargs,
            )

        elif self.robot_controller is not None:

            if robot_id:

                result = await self._call_module(
                    self.robot_controller,
                    (
                        "emergency_stop",
                        "stop",
                        "halt",
                    ),
                    robot_id=robot_id,
                    reason=reason,
                    **kwargs,
                )

            else:

                result = await self._call_module(
                    self.robot_controller,
                    (
                        "emergency_stop_all",
                        "stop_all",
                        "emergency_stop",
                    ),
                    reason=reason,
                    **kwargs,
                )

        await self._emit_event(
            "robotics_emergency_stop",
            {
                "robot_id": robot_id,
                "reason": reason,
                "timestamp": self._timestamp(),
            },
        )

        return {
            "success": True,
            "action": "emergency_stop",
            "robot_id": robot_id,
            "active": True,
            "reason": reason,
            "result": result,
        }

    async def clear_emergency_stop(
        self,
        require_confirmation: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Clear the internal emergency-stop state.

        A confirmation is required by default.
        """

        if require_confirmation:

            confirmed = await self._request_confirmation(
                {
                    "action": "clear_emergency_stop"
                }
            )

            if not confirmed:

                return {
                    "success": False,
                    "action": "clear_emergency_stop",
                    "cancelled": True,
                }

        if self.safety is not None:

            await self._call_module(
                self.safety,
                (
                    "clear_emergency_stop",
                    "reset_emergency_stop",
                    "reset",
                ),
                **kwargs,
            )

        self._emergency_stopped = False

        await self._emit_event(
            "robotics_emergency_stop_cleared",
            {
                "timestamp": self._timestamp(),
            },
        )

        return {
            "success": True,
            "action": "clear_emergency_stop",
            "active": False,
        }

    # ======================================================================
    # SERVO CONTROL
    # ======================================================================

    async def control_servo(
        self,
        robot_id: Optional[str] = None,
        servo_id: Optional[Any] = None,
        angle: Optional[float] = None,
        position: Optional[float] = None,
        speed: Optional[float] = None,
        duration: Optional[float] = None,
        require_confirmation: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Control a servo attached to a robot.
        """

        if not robot_id:

            return self._failure(
                "robot_id is required"
            )

        if servo_id is None:

            return self._failure(
                "servo_id is required"
            )

        if (
            angle is None
            and position is None
        ):

            return self._failure(
                "angle or position is required"
            )

        if self._emergency_stopped:

            return self._failure(
                (
                    "Servo control blocked because "
                    "the emergency stop is active."
                )
            )

        if require_confirmation:

            confirmed = await self._request_confirmation(
                {
                    "action": "servo",
                    "robot_id": robot_id,
                    "servo_id": servo_id,
                    "angle": angle,
                    "position": position,
                    "speed": speed,
                }
            )

            if not confirmed:

                return {
                    "success": False,
                    "action": "servo",
                    "cancelled": True,
                }

        safety_result = await self._check_safety(
            action="servo",
            robot_id=robot_id,
            servo_id=servo_id,
            angle=angle,
            position=position,
            speed=speed,
        )

        if not safety_result["safe"]:

            return {
                "success": False,
                "action": "servo",
                "robot_id": robot_id,
                "servo_id": servo_id,
                "blocked_by_safety": True,
                "reason": safety_result[
                    "reason"
                ],
            }

        if self.servo_controller is not None:

            result = await self._call_module(
                self.servo_controller,
                (
                    "set_angle",
                    "set_position",
                    "move_servo",
                    "control",
                    "set",
                ),
                robot_id=robot_id,
                servo_id=servo_id,
                angle=angle,
                position=position,
                speed=speed,
                duration=duration,
                **kwargs,
            )

            if result is not None:

                self._servo_state[
                    f"{robot_id}:{servo_id}"
                ] = {
                    "angle": angle,
                    "position": position,
                    "speed": speed,
                    "timestamp": self._timestamp(),
                }

                await self._record_command(
                    "servo",
                    {
                        "robot_id": robot_id,
                        "servo_id": servo_id,
                        "angle": angle,
                        "position": position,
                        "speed": speed,
                    },
                )

                return {
                    "success": True,
                    "action": "servo",
                    "robot_id": robot_id,
                    "servo_id": servo_id,
                    "result": result,
                }

        key = (
            f"{robot_id}:{servo_id}"
        )

        self._servo_state[
            key
        ] = {
            "angle": angle,
            "position": position,
            "speed": speed,
            "duration": duration,
            "timestamp": self._timestamp(),
        }

        await self._record_command(
            "servo",
            {
                "robot_id": robot_id,
                "servo_id": servo_id,
                "angle": angle,
                "position": position,
                "speed": speed,
            },
        )

        return {
            "success": True,
            "action": "servo",
            "robot_id": robot_id,
            "servo_id": servo_id,
            "angle": angle,
            "position": position,
            "speed": speed,
            "method": "fallback",
        }

    # ======================================================================
    # SENSOR HANDLING
    # ======================================================================

    async def read_sensor(
        self,
        robot_id: Optional[str] = None,
        sensor_id: Optional[Any] = None,
        sensor_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Read a single robot sensor.
        """

        if not robot_id:

            return self._failure(
                "robot_id is required"
            )

        if sensor_id is None and not sensor_type:

            return self._failure(
                "sensor_id or sensor_type is required"
            )

        if self.sensor_manager is not None:

            result = await self._call_module(
                self.sensor_manager,
                (
                    "read",
                    "read_sensor",
                    "get_sensor",
                    "get_value",
                ),
                robot_id=robot_id,
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                **kwargs,
            )

            if result is not None:

                cache_key = (
                    f"{robot_id}:"
                    f"{sensor_id or sensor_type}"
                )

                self._sensor_cache[
                    cache_key
                ] = result

                return {
                    "success": True,
                    "action": "sensor",
                    "robot_id": robot_id,
                    "sensor_id": sensor_id,
                    "sensor_type": sensor_type,
                    "value": result,
                }

        cache_key = (
            f"{robot_id}:"
            f"{sensor_id or sensor_type}"
        )

        return {
            "success": True,
            "action": "sensor",
            "robot_id": robot_id,
            "sensor_id": sensor_id,
            "sensor_type": sensor_type,
            "value": self._sensor_cache.get(
                cache_key
            ),
            "cached": True,
        }

    async def read_sensors(
        self,
        robot_id: Optional[str] = None,
        sensor_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Read all available sensors or sensors of a requested type.
        """

        if not robot_id:

            return self._failure(
                "robot_id is required"
            )

        if self.sensor_manager is not None:

            result = await self._call_module(
                self.sensor_manager,
                (
                    "read_all",
                    "read_sensors",
                    "get_all",
                    "status",
                ),
                robot_id=robot_id,
                sensor_type=sensor_type,
                **kwargs,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "sensors",
                    "robot_id": robot_id,
                    "sensors": result,
                }

        prefix = (
            f"{robot_id}:"
        )

        cached = {
            key[len(prefix):]: value
            for key, value in (
                self._sensor_cache.items()
            )
            if key.startswith(
                prefix
            )
        }

        return {
            "success": True,
            "action": "sensors",
            "robot_id": robot_id,
            "sensors": cached,
            "cached": True,
        }

    # ======================================================================
    # ROBOT VISION
    # ======================================================================

    async def process_vision(
        self,
        robot_id: Optional[str] = None,
        image: Any = None,
        frame: Any = None,
        operation: str = "analyze",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Send an image/frame to the robotics vision subsystem.
        """

        if not robot_id:

            return self._failure(
                "robot_id is required"
            )

        source = (
            frame
            if frame is not None
            else image
        )

        if self.robot_vision is None:

            return {
                "success": False,
                "action": "vision",
                "robot_id": robot_id,
                "error": (
                    "Robot vision subsystem is "
                    "not available."
                ),
            }

        result = await self._call_module(
            self.robot_vision,
            (
                "analyze",
                "process",
                "detect",
                "recognize",
                "process_frame",
            ),
            robot_id=robot_id,
            image=source,
            frame=source,
            operation=operation,
            **kwargs,
        )

        if result is None:

            return {
                "success": False,
                "action": "vision",
                "robot_id": robot_id,
                "error": (
                    "Robot vision operation "
                    "could not be executed."
                ),
            }

        return {
            "success": True,
            "action": "vision",
            "robot_id": robot_id,
            "operation": operation,
            "result": result,
        }

    # ======================================================================
    # NAVIGATION
    # ======================================================================

    async def navigate(
        self,
        robot_id: Optional[str] = None,
        destination: Any = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        orientation: Optional[float] = None,
        speed: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Navigate a robot toward a destination.
        """

        if not robot_id:

            return self._failure(
                "robot_id is required"
            )

        if (
            destination is None
            and x is None
            and y is None
        ):

            return self._failure(
                "destination or coordinates are required"
            )

        if self._emergency_stopped:

            return self._failure(
                (
                    "Navigation blocked because "
                    "the emergency stop is active."
                )
            )

        safety_result = await self._check_safety(
            action="navigate",
            robot_id=robot_id,
            destination=destination,
            x=x,
            y=y,
            z=z,
            speed=speed,
        )

        if not safety_result["safe"]:

            return {
                "success": False,
                "action": "navigate",
                "robot_id": robot_id,
                "blocked_by_safety": True,
                "reason": safety_result[
                    "reason"
                ],
            }

        if self.navigation is None:

            return {
                "success": False,
                "action": "navigate",
                "robot_id": robot_id,
                "error": (
                    "Navigation subsystem is "
                    "not available."
                ),
            }

        result = await self._call_module(
            self.navigation,
            (
                "navigate",
                "go_to",
                "move_to",
                "goto",
            ),
            robot_id=robot_id,
            destination=destination,
            x=x,
            y=y,
            z=z,
            orientation=orientation,
            speed=speed,
            **kwargs,
        )

        if result is None:

            return {
                "success": False,
                "action": "navigate",
                "robot_id": robot_id,
                "error": (
                    "Navigation operation "
                    "could not be executed."
                ),
            }

        await self._record_command(
            "navigate",
            {
                "robot_id": robot_id,
                "destination": destination,
                "x": x,
                "y": y,
                "z": z,
                "speed": speed,
            },
        )

        return {
            "success": True,
            "action": "navigate",
            "robot_id": robot_id,
            "result": result,
        }

    # ======================================================================
    # OBJECT MANIPULATION
    # ======================================================================

    async def manipulate_object(
        self,
        robot_id: Optional[str] = None,
        operation: Optional[str] = None,
        object_id: Optional[str] = None,
        position: Any = None,
        force: Optional[float] = None,
        require_confirmation: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Perform an object-manipulation operation.

        Examples:

            grab
            release
            pick_up
            place
            move_object
        """

        if not robot_id:

            return self._failure(
                "robot_id is required"
            )

        if not operation:

            return self._failure(
                "operation is required"
            )

        if self._emergency_stopped:

            return self._failure(
                (
                    "Object manipulation blocked "
                    "because the emergency stop is active."
                )
            )

        if require_confirmation:

            confirmed = await self._request_confirmation(
                {
                    "action": "manipulate",
                    "robot_id": robot_id,
                    "operation": operation,
                    "object_id": object_id,
                    "position": position,
                }
            )

            if not confirmed:

                return {
                    "success": False,
                    "action": "manipulate",
                    "robot_id": robot_id,
                    "cancelled": True,
                }

        safety_result = await self._check_safety(
            action="manipulate",
            robot_id=robot_id,
            operation=operation,
            object_id=object_id,
            position=position,
            force=force,
        )

        if not safety_result["safe"]:

            return {
                "success": False,
                "action": "manipulate",
                "robot_id": robot_id,
                "blocked_by_safety": True,
                "reason": safety_result[
                    "reason"
                ],
            }

        if self.object_manipulation is None:

            return {
                "success": False,
                "action": "manipulate",
                "robot_id": robot_id,
                "error": (
                    "Object manipulation subsystem "
                    "is not available."
                ),
            }

        result = await self._call_module(
            self.object_manipulation,
            (
                "manipulate",
                "execute",
                "control",
                operation,
            ),
            robot_id=robot_id,
            operation=operation,
            object_id=object_id,
            position=position,
            force=force,
            **kwargs,
        )

        if result is None:

            return {
                "success": False,
                "action": "manipulate",
                "robot_id": robot_id,
                "error": (
                    "Object manipulation operation "
                    "could not be executed."
                ),
            }

        await self._record_command(
            "manipulate",
            {
                "robot_id": robot_id,
                "operation": operation,
                "object_id": object_id,
                "position": position,
            },
        )

        return {
            "success": True,
            "action": "manipulate",
            "robot_id": robot_id,
            "operation": operation,
            "result": result,
        }

    # ======================================================================
    # RAW ROBOT COMMAND
    # ======================================================================

    async def send_robot_command(
        self,
        robot_id: Optional[str] = None,
        command: Optional[str] = None,
        value: Any = None,
        require_confirmation: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Send a lower-level command to a robot controller.
        """

        if not robot_id:

            return self._failure(
                "robot_id is required"
            )

        if not command:

            return self._failure(
                "command is required"
            )

        if self._emergency_stopped:

            return self._failure(
                (
                    "Robot command blocked because "
                    "the emergency stop is active."
                )
            )

        if require_confirmation:

            confirmed = await self._request_confirmation(
                {
                    "action": "robot_command",
                    "robot_id": robot_id,
                    "command": command,
                    "value": value,
                }
            )

            if not confirmed:

                return {
                    "success": False,
                    "action": "command",
                    "robot_id": robot_id,
                    "cancelled": True,
                }

        safety_result = await self._check_safety(
            action="command",
            robot_id=robot_id,
            command=command,
            value=value,
        )

        if not safety_result["safe"]:

            return {
                "success": False,
                "action": "command",
                "robot_id": robot_id,
                "blocked_by_safety": True,
                "reason": safety_result[
                    "reason"
                ],
            }

        result = None

        if self.robot_controller is not None:

            result = await self._call_module(
                self.robot_controller,
                (
                    "send_command",
                    "command",
                    "execute",
                    "control",
                ),
                robot_id=robot_id,
                command=command,
                value=value,
                **kwargs,
            )

        await self._record_command(
            "command",
            {
                "robot_id": robot_id,
                "command": command,
                "value": value,
            },
        )

        return {
            "success": True,
            "action": "command",
            "robot_id": robot_id,
            "command": command,
            "value": value,
            "result": result,
        }

    # ======================================================================
    # SAFETY
    # ======================================================================

    async def safety_status(
        self,
        robot_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Return current robotics safety status.
        """

        if self.safety is not None:

            result = await self._call_module(
                self.safety,
                (
                    "status",
                    "get_status",
                    "safety_status",
                ),
                robot_id=robot_id,
                **kwargs,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "safety",
                    "robot_id": robot_id,
                    "status": result,
                    "emergency_stopped": (
                        self._emergency_stopped
                    ),
                }

        return {
            "success": True,
            "action": "safety",
            "robot_id": robot_id,
            "status": "operational"
            if not self._emergency_stopped
            else "emergency_stop",
            "emergency_stopped": (
                self._emergency_stopped
            ),
        }

    async def _check_safety(
        self,
        action: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Ask the safety subsystem whether an action is allowed.
        """

        if self._emergency_stopped:

            return {
                "safe": False,
                "reason": (
                    "Global emergency stop is active."
                ),
            }

        if self.safety is None:

            # Without a safety module, basic internal checks still apply.
            speed = kwargs.get(
                "speed"
            )

            if speed is not None:

                try:

                    if float(speed) < 0:

                        return {
                            "safe": False,
                            "reason": (
                                "Speed cannot be negative."
                            ),
                        }

                except (
                    TypeError,
                    ValueError,
                ):

                    return {
                        "safe": False,
                        "reason": (
                            "Invalid speed value."
                        ),
                    }

            return {
                "safe": True,
                "reason": "Basic safety checks passed.",
            }

        result = await self._call_module(
            self.safety,
            (
                "check",
                "check_safety",
                "validate",
                "is_safe",
            ),
            action=action,
            **kwargs,
        )

        if result is None:

            return {
                "safe": True,
                "reason": (
                    "Safety module returned no blocking result."
                ),
            }

        if isinstance(
            result,
            bool,
        ):

            return {
                "safe": result,
                "reason": (
                    "Safety module approved the action."
                    if result
                    else "Safety module blocked the action."
                ),
            }

        if isinstance(
            result,
            dict,
        ):

            safe = result.get(
                "safe",
                result.get(
                    "allowed",
                    True,
                ),
            )

            return {
                "safe": bool(
                    safe
                ),
                "reason": result.get(
                    "reason",
                    "",
                ),
                "details": result,
            }

        return {
            "safe": bool(
                result
            ),
            "reason": str(
                result
            ),
        }

    # ======================================================================
    # CONFIRMATION
    # ======================================================================

    async def _request_confirmation(
        self,
        payload: Dict[str, Any],
    ) -> bool:
        """
        Request confirmation for a potentially physical-risk action.
        """

        if self.confirmation_callback is None:

            # Fail closed when confirmation was explicitly requested.
            return False

        try:

            result = self.confirmation_callback(
                payload
            )

            if asyncio.iscoroutine(
                result
            ):
                result = await result

            return bool(
                result
            )

        except Exception:

            self.logger.exception(
                "Robotics confirmation callback failed."
            )

            return False

    # ======================================================================
    # COMMAND HISTORY
    # ======================================================================

    async def _record_command(
        self,
        command_type: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        Record a robotics command.
        """

        record = {
            "timestamp": self._timestamp(),
            "type": command_type,
            **payload,
        }

        self._command_history.append(
            record
        )

        # Prevent unbounded growth.
        if len(
            self._command_history
        ) > 1000:

            del self._command_history[
                :-1000
            ]

        await self._emit_event(
            "robot_command_recorded",
            record,
        )

    # ======================================================================
    # EVENT SYSTEM
    # ======================================================================

    async def _emit_event(
        self,
        event_name: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        Emit an event to the RENIX event system.
        """

        if self.event_callback is None:
            return

        try:

            result = self.event_callback(
                event_name,
                payload,
            )

            if asyncio.iscoroutine(
                result
            ):
                await result

        except Exception:

            self.logger.exception(
                "Robotics event callback failed."
            )

    # ======================================================================
    # MODULE INVOCATION
    # ======================================================================

    async def _call_module(
        self,
        module: Any,
        method_names: Iterable[str],
        **kwargs: Any,
    ) -> Any:
        """
        Call the first compatible method from a robotics subsystem module.

        Supports both:

            method(**kwargs)

        and:

            method(kwargs)

        because individual RENIX subsystem implementations may expose
        different method signatures.
        """

        if module is None:
            return None

        for method_name in method_names:

            method = getattr(
                module,
                method_name,
                None,
            )

            if not callable(
                method
            ):
                continue

            try:

                result = method(
                    **kwargs
                )

                if asyncio.iscoroutine(
                    result
                ):
                    result = await result

                return result

            except TypeError:

                try:

                    result = method(
                        kwargs
                    )

                    if asyncio.iscoroutine(
                        result
                    ):
                        result = await result

                    return result

                except Exception as exc:

                    self.logger.debug(
                        "Robotics method %s failed: %s",
                        method_name,
                        exc,
                    )

            except Exception as exc:

                self.logger.debug(
                    "Robotics method %s failed: %s",
                    method_name,
                    exc,
                )

        return None

    # ======================================================================
    # NORMALIZATION
    # ======================================================================

    @staticmethod
    def _normalize_robots(
        result: Any,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Normalize different discovery-result formats.
        """

        if result is None:
            return []

        if isinstance(
            result,
            dict,
        ):

            devices = result.get(
                "robots"
            )

            if devices is None:

                devices = result.get(
                    "devices"
                )

            if devices is None:

                return [
                    dict(
                        result
                    )
                ]

            result = devices

        if not isinstance(
            result,
            Sequence,
        ):

            return []

        normalized: List[
            Dict[str, Any]
        ] = []

        for item in result:

            if isinstance(
                item,
                dict,
            ):

                normalized.append(
                    dict(
                        item
                    )
                )

            else:

                normalized.append(
                    {
                        "id": str(
                            item
                        ),
                        "name": str(
                            item
                        ),
                    }
                )

        return normalized

    @staticmethod
    def _robot_id(
        robot: Dict[str, Any],
    ) -> str:
        """
        Extract a stable robot ID.
        """

        value = (
            robot.get("id")
            or robot.get("robot_id")
            or robot.get("uuid")
            or robot.get("serial")
            or robot.get("name")
        )

        if value is None:

            value = (
                "robot-"
                f"{abs(hash(str(robot)))}"
            )

        return str(
            value
        )

    # ======================================================================
    # GENERAL HELPERS
    # ======================================================================

    @staticmethod
    def _timestamp() -> str:
        """
        Return an ISO-8601 UTC timestamp.
        """

        return (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

    @staticmethod
    def _failure(
        message: str,
    ) -> Dict[str, Any]:
        """
        Return a standardized failure response.
        """

        return {
            "success": False,
            "error": message,
        }

    # ======================================================================
    # PUBLIC INFORMATION
    # ======================================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return agent-level status information.
        """

        loaded_modules = []

        module_names = (
            "robot_manager",
            "robot_controller",
            "servo_controller",
            "sensor_manager",
            "robot_vision",
            "navigation",
            "object_manipulation",
            "safety",
        )

        for module_name in module_names:

            if getattr(
                self,
                module_name,
                None,
            ) is not None:

                loaded_modules.append(
                    module_name
                )

        return {
            "agent": self.agent_name,
            "agent_id": self.agent_id,
            "version": self.agent_version,
            "enabled": self.enabled,
            "initialized": self.initialized,
            "known_robots": len(
                self._robots
            ),
            "connected_robots": len(
                self._connections
            ),
            "commands_recorded": len(
                self._command_history
            ),
            "emergency_stopped": (
                self._emergency_stopped
            ),
            "loaded_modules": loaded_modules,
        }

    def get_command_history(
        self,
        limit: int = 100,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Return recent robotics commands.
        """

        if limit <= 0:
            return []

        return list(
            self._command_history[
                -limit:
            ]
        )

    def get_servo_state(
        self,
        robot_id: Optional[str] = None,
        servo_id: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Return cached servo state.
        """

        if robot_id is None:

            return dict(
                self._servo_state
            )

        if servo_id is None:

            prefix = (
                f"{robot_id}:"
            )

            return {
                key[len(prefix):]: value
                for key, value in (
                    self._servo_state.items()
                )
                if key.startswith(
                    prefix
                )
            }

        key = (
            f"{robot_id}:{servo_id}"
        )

        return self._servo_state.get(
            key,
            {},
        )


__all__ = [
    "RoboticsAgent",
]


