"""
RENIX Robotics Object Manipulation
==================================

High-level object manipulation system for RENIX robots.

Features:
- Pick objects
- Place objects
- Grip control
- Release control
- Reach position calculation
- Object selection
- Vision integration
- Safety checks
- Manipulation state tracking
- Task queue
- Event system

Designed to integrate with:
- robot_controller.py
- robot_vision.py
- sensor_manager.py
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(
    "RENIX.Robotics.ObjectManipulation"
)


class ObjectManipulation:
    """
    High-level robotic object manipulation system.

    The actual robotic arm implementation is delegated
    to a robot controller or arm controller.
    """

    def __init__(
        self,
        robot_controller: Optional[Any] = None,
        robot_vision: Optional[Any] = None,
        sensor_manager: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.robot_controller = robot_controller
        self.robot_vision = robot_vision
        self.sensor_manager = sensor_manager

        self.config = config or {}

        self.current_object: Optional[
            Dict[str, Any]
        ] = None

        self.held_object: Optional[
            Dict[str, Any]
        ] = None

        self.gripper_open = True

        self.arm_busy = False

        self.manipulation_enabled = True

        self.max_object_weight = float(
            self.config.get(
                "max_object_weight",
                5.0,
            )
        )

        self.grip_force = float(
            self.config.get(
                "grip_force",
                0.5,
            )
        )

        self.safe_distance = float(
            self.config.get(
                "safe_distance",
                10.0,
            )
        )

        self.task_queue: List[
            Dict[str, Any]
        ] = []

        self.task_history: List[
            Dict[str, Any]
        ] = []

        self.event_handlers: List[
            Callable[[Dict[str, Any]], None]
        ] = []

        self._lock = threading.RLock()

        logger.info(
            "ObjectManipulation initialized."
        )

    # ============================================================
    # OBJECT SELECTION
    # ============================================================

    def select_object(
        self,
        object_data: Dict[str, Any],
    ) -> bool:
        """
        Select an object for manipulation.
        """

        if not object_data:
            return False

        with self._lock:

            self.current_object = dict(
                object_data
            )

        self._emit_event(
            "object_selected",
            object=self.current_object,
        )

        logger.info(
            "Object selected: %s",
            object_data.get(
                "label",
                "unknown",
            ),
        )

        return True

    def select_object_by_label(
        self,
        label: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Select object using robot vision.
        """

        if self.robot_vision is None:
            return None

        try:

            target = (
                self.robot_vision
                .select_target_by_label(
                    label
                )
            )

            if target:

                self.select_object(
                    target
                )

                return target

        except Exception as error:

            logger.error(
                "Object selection failed: %s",
                error,
            )

        return None

    def get_current_object(
        self,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """Return currently selected object."""

        with self._lock:

            if self.current_object is None:
                return None

            return dict(
                self.current_object
            )

    def clear_current_object(
        self,
    ) -> None:
        """Clear selected object."""

        with self._lock:

            self.current_object = None

        self._emit_event(
            "object_cleared"
        )

    # ============================================================
    # PICK OBJECT
    # ============================================================

    def pick_object(
        self,
        object_data: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Pick an object.

        Process:
        1. Validate object
        2. Check safety
        3. Move arm toward object
        4. Open gripper
        5. Reach object
        6. Close gripper
        7. Lift object
        """

        if not self.manipulation_enabled:

            return {
                "success": False,
                "error": (
                    "Object manipulation "
                    "is disabled."
                ),
            }

        if self.arm_busy:

            return {
                "success": False,
                "error": "Robotic arm is busy.",
            }

        if self.held_object is not None:

            return {
                "success": False,
                "error": (
                    "Robot is already holding "
                    "an object."
                ),
            }

        if object_data is None:

            object_data = (
                self.current_object
            )

        if object_data is None:

            return {
                "success": False,
                "error": (
                    "No object selected."
                ),
            }

        if not self._validate_object(
            object_data
        ):

            return {
                "success": False,
                "error": (
                    "Object failed validation."
                ),
            }

        self.arm_busy = True

        try:

            self._emit_event(
                "pick_started",
                object=object_data,
            )

            # Open gripper
            if not self.open_gripper():

                raise RuntimeError(
                    "Failed to open gripper."
                )

            # Calculate object position
            position = (
                self.get_object_position(
                    object_data
                )
            )

            # Move arm to object
            if not self.move_to_position(
                position
            ):

                raise RuntimeError(
                    "Failed to reach object."
                )

            # Close gripper
            if not self.close_gripper():

                raise RuntimeError(
                    "Failed to grip object."
                )

            # Lift object
            if not self.lift_object():

                raise RuntimeError(
                    "Failed to lift object."
                )

            with self._lock:

                self.held_object = dict(
                    object_data
                )

            result = {
                "success": True,
                "action": "picked",
                "object": object_data,
            }

            self._add_history(
                "object_picked",
                object=object_data,
            )

            self._emit_event(
                "object_picked",
                object=object_data,
            )

            logger.info(
                "Object picked successfully: %s",
                object_data.get(
                    "label",
                    "unknown",
                ),
            )

            return result

        except Exception as error:

            logger.error(
                "Pick object failed: %s",
                error,
            )

            self.open_gripper()

            return {
                "success": False,
                "error": str(error),
            }

        finally:

            self.arm_busy = False

    # ============================================================
    # PLACE OBJECT
    # ============================================================

    def place_object(
        self,
        position: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Place currently held object at a position.
        """

        if self.held_object is None:

            return {
                "success": False,
                "error": (
                    "No object is currently held."
                ),
            }

        if self.arm_busy:

            return {
                "success": False,
                "error": "Robotic arm is busy.",
            }

        self.arm_busy = True

        object_data = dict(
            self.held_object
        )

        try:

            self._emit_event(
                "place_started",
                object=object_data,
                position=position,
            )

            if not self.move_to_position(
                position
            ):

                raise RuntimeError(
                    "Failed to reach placement position."
                )

            if not self.open_gripper():

                raise RuntimeError(
                    "Failed to release object."
                )

            with self._lock:

                self.held_object = None

            result = {
                "success": True,
                "action": "placed",
                "object": object_data,
                "position": position,
            }

            self._add_history(
                "object_placed",
                object=object_data,
                position=position,
            )

            self._emit_event(
                "object_placed",
                object=object_data,
                position=position,
            )

            return result

        except Exception as error:

            logger.error(
                "Place object failed: %s",
                error,
            )

            return {
                "success": False,
                "error": str(error),
            }

        finally:

            self.arm_busy = False

    # ============================================================
    # GRIPPER CONTROL
    # ============================================================

    def open_gripper(
        self,
    ) -> bool:
        """Open robotic gripper."""

        try:

            if (
                self.robot_controller
                is not None
                and hasattr(
                    self.robot_controller,
                    "open_gripper",
                )
            ):

                result = (
                    self.robot_controller
                    .open_gripper()
                )

                if result is False:
                    return False

            self.gripper_open = True

            self._emit_event(
                "gripper_opened"
            )

            return True

        except Exception as error:

            logger.error(
                "Failed to open gripper: %s",
                error,
            )

            return False

    def close_gripper(
        self,
        force: Optional[
            float
        ] = None,
    ) -> bool:
        """Close robotic gripper."""

        if force is None:

            force = self.grip_force

        try:

            if (
                self.robot_controller
                is not None
                and hasattr(
                    self.robot_controller,
                    "close_gripper",
                )
            ):

                result = (
                    self.robot_controller
                    .close_gripper(
                        force=force
                    )
                )

                if result is False:
                    return False

            self.gripper_open = False

            self._emit_event(
                "gripper_closed",
                force=force,
            )

            return True

        except Exception as error:

            logger.error(
                "Failed to close gripper: %s",
                error,
            )

            return False

    def release_object(
        self,
    ) -> bool:
        """Release currently held object."""

        if self.held_object is None:

            return False

        success = (
            self.open_gripper()
        )

        if success:

            released = dict(
                self.held_object
            )

            with self._lock:

                self.held_object = None

            self._emit_event(
                "object_released",
                object=released,
            )

        return success

    # ============================================================
    # ARM MOVEMENT
    # ============================================================

    def move_to_position(
        self,
        position: Dict[str, float],
    ) -> bool:
        """
        Move robotic arm to a 3D position.

        Expected position format:

        {
            "x": 10,
            "y": 20,
            "z": 30
        }
        """

        if not self._validate_position(
            position
        ):

            return False

        try:

            if (
                self.robot_controller
                is not None
                and hasattr(
                    self.robot_controller,
                    "move_arm_to",
                )
            ):

                result = (
                    self.robot_controller
                    .move_arm_to(
                        x=position["x"],
                        y=position["y"],
                        z=position["z"],
                    )
                )

                if result is False:
                    return False

            self._emit_event(
                "arm_moved",
                position=position,
            )

            return True

        except Exception as error:

            logger.error(
                "Arm movement failed: %s",
                error,
            )

            return False

    def lift_object(
        self,
        distance: float = 10.0,
    ) -> bool:
        """
        Lift held object upward.
        """

        try:

            if (
                self.robot_controller
                is not None
                and hasattr(
                    self.robot_controller,
                    "lift_arm",
                )
            ):

                result = (
                    self.robot_controller
                    .lift_arm(
                        distance=distance
                    )
                )

                if result is False:
                    return False

            self._emit_event(
                "object_lifted",
                distance=distance,
            )

            return True

        except Exception as error:

            logger.error(
                "Object lift failed: %s",
                error,
            )

            return False

    def move_home(
        self,
    ) -> bool:
        """Move robotic arm to home position."""

        try:

            if (
                self.robot_controller
                is not None
                and hasattr(
                    self.robot_controller,
                    "move_arm_home",
                )
            ):

                result = (
                    self.robot_controller
                    .move_arm_home()
                )

                if result is False:
                    return False

            self._emit_event(
                "arm_home"
            )

            return True

        except Exception as error:

            logger.error(
                "Move home failed: %s",
                error,
            )

            return False

    # ============================================================
    # OBJECT POSITION
    # ============================================================

    def get_object_position(
        self,
        object_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Get estimated 3D object position.

        If the vision system already provides
        coordinates, they are used directly.
        """

        if all(
            key in object_data
            for key in (
                "x",
                "y",
                "z",
            )
        ):

            return {
                "x": float(
                    object_data["x"]
                ),
                "y": float(
                    object_data["y"]
                ),
                "z": float(
                    object_data["z"]
                ),
            }

        metadata = (
            object_data.get(
                "metadata",
                {},
            )
        )

        position = metadata.get(
            "position"
        )

        if (
            isinstance(
                position,
                dict,
            )
            and all(
                key in position
                for key in (
                    "x",
                    "y",
                    "z",
                )
            )
        ):

            return {
                "x": float(
                    position["x"]
                ),
                "y": float(
                    position["y"]
                ),
                "z": float(
                    position["z"]
                ),
            }

        # Fallback estimation from vision data

        distance = object_data.get(
            "distance",
            50.0,
        )

        bbox = object_data.get(
            "bbox"
        )

        if (
            bbox
            and len(bbox) == 4
        ):

            x, y, width, height = bbox

            center_x = (
                x + width / 2
            )

            center_y = (
                y + height / 2
            )

            return {
                "x": float(
                    center_x
                ),
                "y": float(
                    center_y
                ),
                "z": float(
                    distance
                ),
            }

        return {
            "x": 0.0,
            "y": 0.0,
            "z": float(distance),
        }

    # ============================================================
    # SAFETY
    # ============================================================

    def _validate_object(
        self,
        object_data: Dict[str, Any],
    ) -> bool:
        """Validate whether object can be handled."""

        if not isinstance(
            object_data,
            dict,
        ):

            return False

        weight = object_data.get(
            "weight"
        )

        if weight is not None:

            try:

                if (
                    float(weight)
                    > self.max_object_weight
                ):

                    logger.warning(
                        "Object exceeds maximum weight."
                    )

                    return False

            except (
                ValueError,
                TypeError,
            ):

                pass

        if object_data.get(
            "unsafe",
            False,
        ):

            logger.warning(
                "Unsafe object detected."
            )

            return False

        return True

    def _validate_position(
        self,
        position: Dict[str, Any],
    ) -> bool:
        """Validate 3D position."""

        if not isinstance(
            position,
            dict,
        ):

            return False

        required_keys = (
            "x",
            "y",
            "z",
        )

        for key in required_keys:

            if key not in position:
                return False

            try:

                float(
                    position[key]
                )

            except (
                ValueError,
                TypeError,
            ):

                return False

        return True

    def emergency_stop(
        self,
    ) -> None:
        """
        Immediately stop manipulation.
        """

        logger.warning(
            "Object manipulation emergency stop."
        )

        self.arm_busy = False

        try:

            if (
                self.robot_controller
                is not None
                and hasattr(
                    self.robot_controller,
                    "emergency_stop",
                )
            ):

                self.robot_controller.emergency_stop()

        except Exception as error:

            logger.error(
                "Emergency stop failed: %s",
                error,
            )

        self._emit_event(
            "manipulation_emergency_stop"
        )

    # ============================================================
    # TASK QUEUE
    # ============================================================

    def add_task(
        self,
        action: str,
        **data: Any,
    ) -> Dict[str, Any]:
        """
        Add manipulation task to queue.

        Supported actions:
        - pick
        - place
        - release
        - home
        """

        task = {
            "id": (
                f"task_"
                f"{int(time.time() * 1000)}"
            ),
            "action": action,
            "data": data,
            "created_at": (
                datetime.now()
                .isoformat()
            ),
        }

        with self._lock:

            self.task_queue.append(
                task
            )

        self._emit_event(
            "task_added",
            task=task,
        )

        return task

    def execute_next_task(
        self,
    ) -> Dict[str, Any]:
        """Execute next task from queue."""

        with self._lock:

            if not self.task_queue:

                return {
                    "success": False,
                    "error": "Task queue is empty.",
                }

            task = self.task_queue.pop(
                0
            )

        action = task["action"]

        data = task["data"]

        try:

            if action == "pick":

                result = self.pick_object(
                    data.get("object")
                )

            elif action == "place":

                result = self.place_object(
                    data.get(
                        "position",
                        {},
                    )
                )

            elif action == "release":

                success = (
                    self.release_object()
                )

                result = {
                    "success": success,
                    "action": "release",
                }

            elif action == "home":

                success = (
                    self.move_home()
                )

                result = {
                    "success": success,
                    "action": "home",
                }

            else:

                result = {
                    "success": False,
                    "error": (
                        f"Unknown action: {action}"
                    ),
                }

        except Exception as error:

            result = {
                "success": False,
                "error": str(error),
            }

        self._add_history(
            "task_executed",
            task=task,
            result=result,
        )

        return result

    def clear_task_queue(
        self,
    ) -> None:
        """Clear all queued tasks."""

        with self._lock:

            self.task_queue.clear()

        self._emit_event(
            "task_queue_cleared"
        )

    # ============================================================
    # HISTORY
    # ============================================================

    def _add_history(
        self,
        event: str,
        **data: Any,
    ) -> None:
        """Add event to manipulation history."""

        record = {
            "event": event,
            "timestamp": (
                datetime.now()
                .isoformat()
            ),
            **data,
        }

        self.task_history.append(
            record
        )

        max_history = int(
            self.config.get(
                "max_history",
                500,
            )
        )

        if (
            len(self.task_history)
            > max_history
        ):

            self.task_history = (
                self.task_history[
                    -max_history:
                ]
            )

    def get_history(
        self,
        limit: int = 100,
    ) -> List[
        Dict[str, Any]
    ]:
        """Return manipulation history."""

        return self.task_history[
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
        """Register event handler."""

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
        """Remove event handler."""

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
        """Emit manipulation event."""

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
                    "Event handler error: %s",
                    error,
                )

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """Return manipulation system status."""

        return {
            "enabled": (
                self.manipulation_enabled
            ),
            "arm_busy": self.arm_busy,
            "gripper_open": (
                self.gripper_open
            ),
            "current_object": (
                self.get_current_object()
            ),
            "held_object": (
                dict(self.held_object)
                if self.held_object
                else None
            ),
            "queued_tasks": len(
                self.task_queue
            ),
            "history_size": len(
                self.task_history
            ),
        }

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """Safely shutdown manipulation system."""

        logger.info(
            "Shutting down ObjectManipulation..."
        )

        self.clear_task_queue()

        if self.held_object is not None:

            self.release_object()

        self.move_home()

        with self._lock:

            self.current_object = None
            self.event_handlers.clear()

        logger.info(
            "ObjectManipulation shutdown complete."
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

        def open_gripper(self):

            print(
                "Gripper opened."
            )

            return True

        def close_gripper(
            self,
            force=0.5,
        ):

            print(
                f"Gripper closed "
                f"with force {force}"
            )

            return True

        def move_arm_to(
            self,
            x,
            y,
            z,
        ):

            print(
                f"Moving arm to "
                f"X={x}, Y={y}, Z={z}"
            )

            return True

        def lift_arm(
            self,
            distance=10,
        ):

            print(
                f"Lifting arm "
                f"{distance} units."
            )

            return True

        def move_arm_home(self):

            print(
                "Moving arm home."
            )

            return True


    manipulation = ObjectManipulation(
        robot_controller=(
            DemoRobotController()
        )
    )

    test_object = {
        "label": "bottle",
        "confidence": 0.95,
        "distance": 30,
        "bbox": (
            100,
            100,
            80,
            200,
        ),
        "weight": 0.5,
    }

    manipulation.select_object(
        test_object
    )

    print("\nPicking Object:")

    print(
        manipulation.pick_object()
    )

    print("\nPlacing Object:")

    print(
        manipulation.place_object(
            {
                "x": 50,
                "y": 30,
                "z": 20,
            }
        )
    )

    print("\nStatus:")

    print(
        manipulation.get_status()
    )

    manipulation.shutdown()


