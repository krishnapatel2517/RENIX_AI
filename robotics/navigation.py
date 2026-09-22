"""
RENIX Robotics Navigation
=========================

High-level navigation system for RENIX robots.

Features:
- Position tracking
- Goal navigation
- Waypoint navigation
- Direction calculation
- Distance calculation
- Obstacle-aware movement
- Path management
- Navigation state
- Emergency stop support
- Integration with RobotVision
- Integration with SensorManager
- Integration with RobotController
"""

from __future__ import annotations

import logging
import math
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger("RENIX.Robotics.Navigation")


class Navigation:
    """
    High-level robot navigation engine.

    Coordinates robot movement between positions and can integrate with:

    - robot_controller.py
    - sensor_manager.py
    - robot_vision.py
    """

    def __init__(
        self,
        robot_controller: Optional[Any] = None,
        sensor_manager: Optional[Any] = None,
        robot_vision: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.robot_controller = robot_controller
        self.sensor_manager = sensor_manager
        self.robot_vision = robot_vision

        self.config = config or {}

        # Current robot position
        self.position = {
            "x": float(
                self.config.get(
                    "start_x",
                    0.0,
                )
            ),
            "y": float(
                self.config.get(
                    "start_y",
                    0.0,
                )
            ),
            "heading": float(
                self.config.get(
                    "start_heading",
                    0.0,
                )
            ),
        }

        self.goal: Optional[
            Dict[str, float]
        ] = None

        self.path: List[
            Dict[str, float]
        ] = []

        self.current_waypoint_index = 0

        self.running = False
        self.paused = False

        self.navigation_thread: Optional[
            threading.Thread
        ] = None

        self._stop_event = threading.Event()

        self._lock = threading.RLock()

        self.event_handlers: List[
            Callable[[Dict[str, Any]], None]
        ] = []

        # Navigation configuration
        self.position_tolerance = float(
            self.config.get(
                "position_tolerance",
                10.0,
            )
        )

        self.obstacle_distance = float(
            self.config.get(
                "obstacle_distance",
                50.0,
            )
        )

        self.move_speed = float(
            self.config.get(
                "move_speed",
                0.5,
            )
        )

        self.turn_speed = float(
            self.config.get(
                "turn_speed",
                0.4,
            )
        )

        self.update_interval = float(
            self.config.get(
                "update_interval",
                0.1,
            )
        )

        self.last_navigation_action = "idle"

        self.navigation_history: List[
            Dict[str, Any]
        ] = []

        logger.info(
            "Navigation system initialized."
        )

    # ============================================================
    # POSITION MANAGEMENT
    # ============================================================

    def set_position(
        self,
        x: float,
        y: float,
        heading: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Update robot position.

        Args:
            x: X coordinate
            y: Y coordinate
            heading: Optional robot direction in degrees
        """

        with self._lock:

            self.position["x"] = float(x)
            self.position["y"] = float(y)

            if heading is not None:

                self.position[
                    "heading"
                ] = self._normalize_angle(
                    float(heading)
                )

        self._emit_event(
            "position_updated",
            position=self.get_position(),
        )

        return self.get_position()

    def update_position(
        self,
        distance: float,
        heading: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Update position based on movement distance.

        Useful when integrating with wheel encoders
        or odometry systems.
        """

        with self._lock:

            if heading is None:

                heading = self.position[
                    "heading"
                ]

            radians = math.radians(
                heading
            )

            self.position["x"] += (
                distance
                * math.cos(radians)
            )

            self.position["y"] += (
                distance
                * math.sin(radians)
            )

            self.position[
                "heading"
            ] = self._normalize_angle(
                heading
            )

        return self.get_position()

    def get_position(
        self,
    ) -> Dict[str, float]:
        """Return current robot position."""

        with self._lock:

            return dict(
                self.position
            )

    # ============================================================
    # GOAL MANAGEMENT
    # ============================================================

    def set_goal(
        self,
        x: float,
        y: float,
    ) -> Dict[str, Any]:
        """
        Set a navigation goal.
        """

        with self._lock:

            self.goal = {
                "x": float(x),
                "y": float(y),
            }

        self._emit_event(
            "goal_set",
            goal=dict(
                self.goal
            ),
        )

        logger.info(
            "Navigation goal set: %s",
            self.goal,
        )

        return {
            "success": True,
            "goal": dict(
                self.goal
            ),
        }

    def clear_goal(
        self,
    ) -> None:
        """Clear current navigation goal."""

        with self._lock:

            self.goal = None

        self._emit_event(
            "goal_cleared"
        )

    def get_goal(
        self,
    ) -> Optional[
        Dict[str, float]
    ]:
        """Return current navigation goal."""

        with self._lock:

            if self.goal is None:
                return None

            return dict(
                self.goal
            )

    def goal_reached(
        self,
    ) -> bool:
        """
        Check whether robot reached goal.
        """

        if self.goal is None:
            return False

        distance = (
            self.calculate_distance(
                self.position,
                self.goal,
            )
        )

        return (
            distance
            <= self.position_tolerance
        )

    # ============================================================
    # WAYPOINTS
    # ============================================================

    def set_path(
        self,
        waypoints: List[
            Dict[str, float]
        ],
    ) -> bool:
        """
        Set navigation path.

        Example:

            [
                {"x": 100, "y": 0},
                {"x": 100, "y": 100},
                {"x": 200, "y": 100}
            ]
        """

        if not waypoints:
            return False

        validated_path = []

        for waypoint in waypoints:

            if (
                "x" not in waypoint
                or "y" not in waypoint
            ):

                return False

            validated_path.append(
                {
                    "x": float(
                        waypoint["x"]
                    ),
                    "y": float(
                        waypoint["y"]
                    ),
                }
            )

        with self._lock:

            self.path = validated_path

            self.current_waypoint_index = 0

        self._emit_event(
            "path_set",
            waypoint_count=len(
                self.path
            ),
        )

        return True

    def add_waypoint(
        self,
        x: float,
        y: float,
    ) -> None:
        """Add waypoint to current path."""

        waypoint = {
            "x": float(x),
            "y": float(y),
        }

        with self._lock:

            self.path.append(
                waypoint
            )

        self._emit_event(
            "waypoint_added",
            waypoint=waypoint,
        )

    def clear_path(
        self,
    ) -> None:
        """Clear navigation path."""

        with self._lock:

            self.path.clear()

            self.current_waypoint_index = 0

        self._emit_event(
            "path_cleared"
        )

    def get_current_waypoint(
        self,
    ) -> Optional[
        Dict[str, float]
    ]:
        """Return current waypoint."""

        with self._lock:

            if (
                self.current_waypoint_index
                >= len(self.path)
            ):

                return None

            return dict(
                self.path[
                    self.current_waypoint_index
                ]
            )

    def advance_waypoint(
        self,
    ) -> bool:
        """
        Move to the next waypoint.
        """

        with self._lock:

            self.current_waypoint_index += 1

            if (
                self.current_waypoint_index
                >= len(self.path)
            ):

                self._emit_event(
                    "path_completed"
                )

                return False

        self._emit_event(
            "waypoint_reached",
            waypoint_index=(
                self.current_waypoint_index
                - 1
            ),
        )

        return True

    # ============================================================
    # NAVIGATION CONTROL
    # ============================================================

    def navigate_to(
        self,
        x: float,
        y: float,
        blocking: bool = False,
        timeout: Optional[
            float
        ] = None,
    ) -> Dict[str, Any]:
        """
        Navigate robot to a position.

        Args:
            x:
                Target X coordinate.

            y:
                Target Y coordinate.

            blocking:
                Wait until destination is reached.

            timeout:
                Maximum navigation time.
        """

        self.set_goal(
            x,
            y,
        )

        self.start_navigation()

        if not blocking:

            return {
                "success": True,
                "status": "navigation_started",
                "goal": self.get_goal(),
            }

        start_time = time.time()

        while self.running:

            if self.goal_reached():

                self.stop_navigation()

                return {
                    "success": True,
                    "status": "goal_reached",
                    "position": (
                        self.get_position()
                    ),
                }

            if (
                timeout is not None
                and (
                    time.time()
                    - start_time
                    > timeout
                )
            ):

                self.stop_navigation()

                return {
                    "success": False,
                    "status": "timeout",
                }

            time.sleep(
                0.1
            )

        return {
            "success": False,
            "status": "navigation_stopped",
        }

    def start_navigation(
        self,
    ) -> bool:
        """Start navigation engine."""

        if self.running:
            return True

        if (
            self.goal is None
            and not self.path
        ):

            logger.warning(
                "No navigation goal or path."
            )

            return False

        self.running = True

        self.paused = False

        self._stop_event.clear()

        self.navigation_thread = (
            threading.Thread(
                target=self._navigation_loop,
                daemon=True,
                name="RENIX-Navigation",
            )
        )

        self.navigation_thread.start()

        self._emit_event(
            "navigation_started"
        )

        logger.info(
            "Navigation started."
        )

        return True

    def stop_navigation(
        self,
    ) -> None:
        """Stop robot navigation."""

        if not self.running:
            return

        self.running = False

        self._stop_event.set()

        self._stop_robot()

        if (
            self.navigation_thread
            and self.navigation_thread.is_alive()
        ):

            self.navigation_thread.join(
                timeout=2
            )

        self.navigation_thread = None

        self.last_navigation_action = (
            "stopped"
        )

        self._emit_event(
            "navigation_stopped"
        )

    def pause_navigation(
        self,
    ) -> None:
        """Pause navigation."""

        if not self.running:
            return

        self.paused = True

        self._stop_robot()

        self.last_navigation_action = (
            "paused"
        )

        self._emit_event(
            "navigation_paused"
        )

    def resume_navigation(
        self,
    ) -> None:
        """Resume paused navigation."""

        if not self.running:
            return

        self.paused = False

        self._emit_event(
            "navigation_resumed"
        )

    # ============================================================
    # NAVIGATION LOOP
    # ============================================================

    def _navigation_loop(
        self,
    ) -> None:
        """Main autonomous navigation loop."""

        while (
            self.running
            and not self._stop_event.is_set()
        ):

            try:

                if self.paused:

                    self._stop_event.wait(
                        self.update_interval
                    )

                    continue

                target = (
                    self._get_navigation_target()
                )

                if target is None:

                    self.stop_navigation()

                    break

                if self._is_obstacle_detected():

                    self._handle_obstacle()

                    self._stop_event.wait(
                        self.update_interval
                    )

                    continue

                distance = (
                    self.calculate_distance(
                        self.position,
                        target,
                    )
                )

                if (
                    distance
                    <= self.position_tolerance
                ):

                    self._handle_target_reached(
                        target
                    )

                else:

                    self._navigate_towards(
                        target
                    )

            except Exception as error:

                logger.error(
                    "Navigation loop error: %s",
                    error,
                )

                self._emit_event(
                    "navigation_error",
                    error=str(error),
                )

            self._stop_event.wait(
                self.update_interval
            )

    # ============================================================
    # TARGET LOGIC
    # ============================================================

    def _get_navigation_target(
        self,
    ) -> Optional[
        Dict[str, float]
    ]:
        """
        Get current navigation target.
        """

        waypoint = (
            self.get_current_waypoint()
        )

        if waypoint is not None:

            return waypoint

        return self.get_goal()

    def _handle_target_reached(
        self,
        target: Dict[str, float],
    ) -> None:
        """
        Handle waypoint or goal completion.
        """

        self._stop_robot()

        waypoint = (
            self.get_current_waypoint()
        )

        if waypoint is not None:

            self._emit_event(
                "waypoint_reached",
                waypoint=waypoint,
            )

            has_next = (
                self.advance_waypoint()
            )

            if has_next:
                return

        if (
            self.goal is not None
            and target == self.goal
        ):

            self._emit_event(
                "goal_reached",
                goal=dict(
                    self.goal
                ),
                position=(
                    self.get_position()
                ),
            )

            self.last_navigation_action = (
                "goal_reached"
            )

            self.running = False

            self._stop_event.set()

    # ============================================================
    # MOVEMENT
    # ============================================================

    def _navigate_towards(
        self,
        target: Dict[str, float],
    ) -> None:
        """
        Calculate direction and move robot
        toward target.
        """

        desired_heading = (
            self.calculate_heading(
                self.position,
                target,
            )
        )

        current_heading = (
            self.position["heading"]
        )

        heading_difference = (
            self.angle_difference(
                current_heading,
                desired_heading,
            )
        )

        if abs(
            heading_difference
        ) > 10:

            if heading_difference > 0:

                self._turn_right()

            else:

                self._turn_left()

        else:

            self._move_forward()

    def _move_forward(
        self,
    ) -> None:
        """Command robot to move forward."""

        self.last_navigation_action = (
            "move_forward"
        )

        try:

            if (
                self.robot_controller
                is not None
                and hasattr(
                    self.robot_controller,
                    "move_forward",
                )
            ):

                self.robot_controller.move_forward(
                    speed=self.move_speed
                )

        except Exception as error:

            logger.error(
                "Move forward failed: %s",
                error,
            )

    def _turn_left(
        self,
    ) -> None:
        """Command robot to turn left."""

        self.last_navigation_action = (
            "turn_left"
        )

        try:

            if (
                self.robot_controller
                is not None
                and hasattr(
                    self.robot_controller,
                    "turn_left",
                )
            ):

                self.robot_controller.turn_left(
                    speed=self.turn_speed
                )

        except Exception as error:

            logger.error(
                "Turn left failed: %s",
                error,
            )

    def _turn_right(
        self,
    ) -> None:
        """Command robot to turn right."""

        self.last_navigation_action = (
            "turn_right"
        )

        try:

            if (
                self.robot_controller
                is not None
                and hasattr(
                    self.robot_controller,
                    "turn_right",
                )
            ):

                self.robot_controller.turn_right(
                    speed=self.turn_speed
                )

        except Exception as error:

            logger.error(
                "Turn right failed: %s",
                error,
            )

    def _stop_robot(
        self,
    ) -> None:
        """Stop robot movement."""

        try:

            if (
                self.robot_controller
                is not None
                and hasattr(
                    self.robot_controller,
                    "stop",
                )
            ):

                self.robot_controller.stop()

        except Exception as error:

            logger.error(
                "Robot stop failed: %s",
                error,
            )

    # ============================================================
    # OBSTACLE DETECTION
    # ============================================================

    def _is_obstacle_detected(
        self,
    ) -> bool:
        """
        Check sensors and vision for obstacles.
        """

        # Sensor manager
        if self.sensor_manager is not None:

            try:

                sensor_result = (
                    self.sensor_manager.read_all()
                )

                results = sensor_result.get(
                    "results",
                    {},
                )

                for result in results.values():

                    if not result.get(
                        "success",
                        False,
                    ):
                        continue

                    value = result.get(
                        "value"
                    )

                    if isinstance(
                        value,
                        (
                            int,
                            float,
                        ),
                    ):

                        if (
                            value
                            <= self.obstacle_distance
                        ):

                            return True

            except Exception as error:

                logger.debug(
                    "Sensor obstacle check failed: %s",
                    error,
                )

        # Robot vision
        if self.robot_vision is not None:

            try:

                obstacles = (
                    self.robot_vision.detect_obstacles()
                )

                if obstacles:

                    return True

            except Exception as error:

                logger.debug(
                    "Vision obstacle check failed: %s",
                    error,
                )

        return False

    def _handle_obstacle(
        self,
    ) -> None:
        """
        Handle detected obstacle.
        """

        self._stop_robot()

        self.last_navigation_action = (
            "obstacle_detected"
        )

        self._emit_event(
            "navigation_obstacle_detected",
            position=self.get_position(),
        )

        logger.warning(
            "Navigation stopped due to obstacle."
        )

    # ============================================================
    # PATH CALCULATIONS
    # ============================================================

    @staticmethod
    def calculate_distance(
        position_a: Dict[str, float],
        position_b: Dict[str, float],
    ) -> float:
        """
        Calculate Euclidean distance
        between two positions.
        """

        dx = (
            float(position_b["x"])
            - float(position_a["x"])
        )

        dy = (
            float(position_b["y"])
            - float(position_a["y"])
        )

        return math.sqrt(
            dx * dx
            + dy * dy
        )

    @staticmethod
    def calculate_heading(
        position_a: Dict[str, float],
        position_b: Dict[str, float],
    ) -> float:
        """
        Calculate required heading
        from position A to B.
        """

        dx = (
            float(position_b["x"])
            - float(position_a["x"])
        )

        dy = (
            float(position_b["y"])
            - float(position_a["y"])
        )

        angle = math.degrees(
            math.atan2(
                dy,
                dx,
            )
        )

        return (
            angle
            + 360
        ) % 360

    @staticmethod
    def angle_difference(
        current: float,
        target: float,
    ) -> float:
        """
        Calculate shortest angle difference.
        """

        difference = (
            target
            - current
            + 180
        ) % 360 - 180

        return difference

    @staticmethod
    def _normalize_angle(
        angle: float,
    ) -> float:
        """Normalize angle to 0-360."""

        return (
            angle
            + 360
        ) % 360

    # ============================================================
    # PATH PLANNING
    # ============================================================

    def create_direct_path(
        self,
        target_x: float,
        target_y: float,
        step_size: float = 50.0,
    ) -> List[
        Dict[str, float]
    ]:
        """
        Create a simple direct waypoint path.

        This is a basic planner. More advanced
        algorithms such as A*, RRT, SLAM, etc.
        can later replace or extend this.
        """

        start = self.get_position()

        target = {
            "x": float(target_x),
            "y": float(target_y),
        }

        distance = (
            self.calculate_distance(
                start,
                target,
            )
        )

        if distance == 0:

            return []

        steps = max(
            1,
            int(
                math.ceil(
                    distance
                    / step_size
                )
            ),
        )

        path = []

        for index in range(
            1,
            steps + 1,
        ):

            ratio = (
                index
                / steps
            )

            waypoint = {
                "x": (
                    start["x"]
                    + (
                        target["x"]
                        - start["x"]
                    )
                    * ratio
                ),
                "y": (
                    start["y"]
                    + (
                        target["y"]
                        - start["y"]
                    )
                    * ratio
                ),
            }

            path.append(
                waypoint
            )

        return path

    # ============================================================
    # HISTORY
    # ============================================================

    def add_history(
        self,
        event: str,
        **data: Any,
    ) -> None:
        """Add navigation event to history."""

        record = {
            "event": event,
            "timestamp": (
                datetime.now()
                .isoformat()
            ),
            "position": (
                self.get_position()
            ),
            **data,
        }

        self.navigation_history.append(
            record
        )

        max_history = int(
            self.config.get(
                "max_history",
                500,
            )
        )

        if (
            len(
                self.navigation_history
            )
            > max_history
        ):

            self.navigation_history = (
                self.navigation_history[
                    -max_history:
                ]
            )

    def get_history(
        self,
        limit: int = 100,
    ) -> List[
        Dict[str, Any]
    ]:
        """Return navigation history."""

        return self.navigation_history[
            -max(
                1,
                int(limit),
            ):
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
        """Add navigation event handler."""

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
        """Remove navigation event handler."""

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
        """Emit navigation event."""

        event = {
            "event": event_name,
            "timestamp": (
                datetime.now()
                .isoformat()
            ),
            **data,
        }

        self.add_history(
            event_name,
            **data,
        )

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
                    "Navigation event error: %s",
                    error,
                )

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """Return navigation system status."""

        return {
            "running": self.running,
            "paused": self.paused,
            "position": (
                self.get_position()
            ),
            "goal": self.get_goal(),
            "waypoint_count": len(
                self.path
            ),
            "current_waypoint_index": (
                self.current_waypoint_index
            ),
            "current_waypoint": (
                self.get_current_waypoint()
            ),
            "last_action": (
                self.last_navigation_action
            ),
        }

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """Safely shutdown navigation."""

        logger.info(
            "Shutting down Navigation..."
        )

        self.stop_navigation()

        self.clear_goal()

        self.clear_path()

        with self._lock:

            self.event_handlers.clear()

        logger.info(
            "Navigation shutdown complete."
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


    class DemoRobotController:

        def move_forward(
            self,
            speed=0.5,
        ):

            print(
                f"Robot moving forward "
                f"at speed {speed}"
            )

        def turn_left(
            self,
            speed=0.4,
        ):

            print(
                f"Robot turning left "
                f"at speed {speed}"
            )

        def turn_right(
            self,
            speed=0.4,
        ):

            print(
                f"Robot turning right "
                f"at speed {speed}"
            )

        def stop(self):

            print(
                "Robot stopped."
            )


    navigation = Navigation(
        robot_controller=(
            DemoRobotController()
        ),
        config={
            "position_tolerance": 10,
            "move_speed": 0.5,
            "turn_speed": 0.4,
        },
    )

    navigation.set_position(
        x=0,
        y=0,
        heading=0,
    )

    print(
        "\nCurrent Position:"
    )

    print(
        navigation.get_position()
    )

    navigation.set_goal(
        200,
        100,
    )

    print(
        "\nGoal:"
    )

    print(
        navigation.get_goal()
    )

    path = (
        navigation.create_direct_path(
            200,
            100,
        )
    )

    print(
        "\nGenerated Path:"
    )

    for waypoint in path:

        print(
            waypoint
        )

    print(
        "\nNavigation Status:"
    )

    print(
        navigation.get_status()
    )

    navigation.shutdown()


