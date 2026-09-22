"""
RENIX Robot Vision
==================

Robot-specific vision intelligence layer.

This module connects the general RENIX vision system with robotics and provides:
- Camera integration
- Object detection
- Person detection
- Face detection
- Object tracking
- Distance estimation
- Target selection
- Spatial awareness
- Obstacle detection
- Scene analysis
- Robot-following support
- Vision event callbacks

Hardware and AI models are intentionally abstracted so different cameras,
OpenCV pipelines, YOLO models, depth cameras, etc. can be plugged in later.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


logger = logging.getLogger("RENIX.Robotics.RobotVision")


class RobotVision:
    """
    High-level robot vision system.

    The class can work with a vision adapter implementing methods such as:

        connect()
        disconnect()
        start()
        stop()
        get_frame()
        detect_objects(frame)
        detect_people(frame)
        detect_faces(frame)
        estimate_depth(frame)
    """

    def __init__(
        self,
        vision_adapter: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.vision_adapter = vision_adapter
        self.config = config or {}

        self.connected = False
        self.running = False

        self.current_frame: Optional[Any] = None

        self.last_analysis: Dict[str, Any] = {}

        self.detected_objects: List[Dict[str, Any]] = []
        self.detected_people: List[Dict[str, Any]] = []
        self.detected_faces: List[Dict[str, Any]] = []

        self.tracked_objects: Dict[str, Dict[str, Any]] = {}

        self.current_target: Optional[Dict[str, Any]] = None

        self.event_handlers: List[
            Callable[[Dict[str, Any]], None]
        ] = []

        self.analysis_history = deque(
            maxlen=int(
                self.config.get(
                    "history_size",
                    100,
                )
            )
        )

        self.confidence_threshold = float(
            self.config.get(
                "confidence_threshold",
                0.5,
            )
        )

        self.obstacle_distance_threshold = float(
            self.config.get(
                "obstacle_distance_threshold",
                100.0,
            )
        )

        self.update_interval = float(
            self.config.get(
                "update_interval",
                0.1,
            )
        )

        self._stop_event = threading.Event()

        self._vision_thread: Optional[
            threading.Thread
        ] = None

        self._lock = threading.RLock()

        logger.info(
            "RobotVision initialized."
        )

    # ============================================================
    # CONNECTION
    # ============================================================

    def connect(self) -> bool:
        """Connect to the vision system."""

        if self.connected:
            return True

        try:

            if (
                self.vision_adapter is not None
                and hasattr(
                    self.vision_adapter,
                    "connect",
                )
            ):

                result = (
                    self.vision_adapter.connect()
                )

                if result is False:
                    return False

            self.connected = True

            self._emit_event(
                "robot_vision_connected"
            )

            logger.info(
                "Robot vision connected."
            )

            return True

        except Exception as error:

            logger.error(
                "Robot vision connection failed: %s",
                error,
            )

            return False

    def disconnect(self) -> bool:
        """Disconnect the vision system."""

        self.stop()

        try:

            if (
                self.vision_adapter is not None
                and hasattr(
                    self.vision_adapter,
                    "disconnect",
                )
            ):

                result = (
                    self.vision_adapter.disconnect()
                )

                if result is False:
                    return False

            self.connected = False

            self._emit_event(
                "robot_vision_disconnected"
            )

            return True

        except Exception as error:

            logger.error(
                "Robot vision disconnect failed: %s",
                error,
            )

            return False

    # ============================================================
    # START / STOP
    # ============================================================

    def start(self) -> bool:
        """Start continuous robot vision."""

        if self.running:
            return True

        if not self.connected:

            if not self.connect():
                return False

        try:

            if (
                self.vision_adapter is not None
                and hasattr(
                    self.vision_adapter,
                    "start",
                )
            ):

                result = (
                    self.vision_adapter.start()
                )

                if result is False:
                    return False

            self._stop_event.clear()

            self.running = True

            self._vision_thread = (
                threading.Thread(
                    target=self._vision_loop,
                    daemon=True,
                    name="RENIX-RobotVision",
                )
            )

            self._vision_thread.start()

            self._emit_event(
                "robot_vision_started"
            )

            logger.info(
                "Robot vision started."
            )

            return True

        except Exception as error:

            logger.error(
                "Failed to start robot vision: %s",
                error,
            )

            self.running = False

            return False

    def stop(self) -> None:
        """Stop continuous robot vision."""

        if not self.running:
            return

        self.running = False

        self._stop_event.set()

        if (
            self._vision_thread is not None
            and self._vision_thread.is_alive()
        ):

            self._vision_thread.join(
                timeout=3
            )

        self._vision_thread = None

        try:

            if (
                self.vision_adapter is not None
                and hasattr(
                    self.vision_adapter,
                    "stop",
                )
            ):

                self.vision_adapter.stop()

        except Exception as error:

            logger.warning(
                "Vision adapter stop error: %s",
                error,
            )

        self._emit_event(
            "robot_vision_stopped"
        )

        logger.info(
            "Robot vision stopped."
        )

    # ============================================================
    # MAIN VISION LOOP
    # ============================================================

    def _vision_loop(self) -> None:
        """Background continuous vision loop."""

        while not self._stop_event.is_set():

            try:

                self.analyze_scene()

            except Exception as error:

                logger.error(
                    "Robot vision loop error: %s",
                    error,
                )

            self._stop_event.wait(
                self.update_interval
            )

    # ============================================================
    # FRAME
    # ============================================================

    def get_frame(self) -> Optional[Any]:
        """Get the latest camera frame."""

        try:

            if (
                self.vision_adapter is not None
                and hasattr(
                    self.vision_adapter,
                    "get_frame",
                )
            ):

                frame = (
                    self.vision_adapter.get_frame()
                )

                self.current_frame = frame

                return frame

        except Exception as error:

            logger.error(
                "Failed to get camera frame: %s",
                error,
            )

        return self.current_frame

    # ============================================================
    # SCENE ANALYSIS
    # ============================================================

    def analyze_scene(
        self,
        frame: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Analyze the current environment.

        Detects:
        - Objects
        - People
        - Faces
        - Obstacles
        - Depth information
        """

        if frame is None:

            frame = self.get_frame()

        if frame is None:

            return {
                "success": False,
                "error": "No camera frame available.",
            }

        objects = self.detect_objects(
            frame
        )

        people = self.detect_people(
            frame
        )

        faces = self.detect_faces(
            frame
        )

        depth_data = self.estimate_depth(
            frame
        )

        obstacles = self.detect_obstacles(
            objects,
            depth_data,
        )

        timestamp = datetime.now().isoformat()

        analysis = {
            "success": True,
            "timestamp": timestamp,
            "objects": objects,
            "people": people,
            "faces": faces,
            "obstacles": obstacles,
            "depth": depth_data,
        }

        with self._lock:

            self.last_analysis = analysis

            self.detected_objects = objects

            self.detected_people = people

            self.detected_faces = faces

            self.analysis_history.append(
                analysis
            )

        self._update_tracking(
            objects
        )

        if obstacles:

            self._emit_event(
                "obstacle_detected",
                obstacles=obstacles,
            )

        self._emit_event(
            "scene_analyzed",
            object_count=len(objects),
            people_count=len(people),
            face_count=len(faces),
        )

        return analysis

    # ============================================================
    # OBJECT DETECTION
    # ============================================================

    def detect_objects(
        self,
        frame: Any,
    ) -> List[Dict[str, Any]]:
        """Detect objects in a frame."""

        objects: List[
            Dict[str, Any]
        ] = []

        try:

            if (
                self.vision_adapter is not None
                and hasattr(
                    self.vision_adapter,
                    "detect_objects",
                )
            ):

                results = (
                    self.vision_adapter.detect_objects(
                        frame
                    )
                )

                for item in results or []:

                    normalized = (
                        self._normalize_detection(
                            item
                        )
                    )

                    if (
                        normalized["confidence"]
                        >= self.confidence_threshold
                    ):

                        objects.append(
                            normalized
                        )

        except Exception as error:

            logger.error(
                "Object detection failed: %s",
                error,
            )

        return objects

    # ============================================================
    # PERSON DETECTION
    # ============================================================

    def detect_people(
        self,
        frame: Any,
    ) -> List[Dict[str, Any]]:
        """Detect people."""

        people: List[
            Dict[str, Any]
        ] = []

        try:

            if (
                self.vision_adapter is not None
                and hasattr(
                    self.vision_adapter,
                    "detect_people",
                )
            ):

                results = (
                    self.vision_adapter.detect_people(
                        frame
                    )
                )

                for person in results or []:

                    normalized = (
                        self._normalize_detection(
                            person,
                            default_label="person",
                        )
                    )

                    if (
                        normalized["confidence"]
                        >= self.confidence_threshold
                    ):

                        people.append(
                            normalized
                        )

            else:

                objects = (
                    self.detect_objects(
                        frame
                    )
                )

                people = [
                    obj
                    for obj in objects
                    if obj[
                        "label"
                    ].lower()
                    == "person"
                ]

        except Exception as error:

            logger.error(
                "Person detection failed: %s",
                error,
            )

        return people

    # ============================================================
    # FACE DETECTION
    # ============================================================

    def detect_faces(
        self,
        frame: Any,
    ) -> List[Dict[str, Any]]:
        """Detect faces."""

        faces: List[
            Dict[str, Any]
        ] = []

        try:

            if (
                self.vision_adapter is not None
                and hasattr(
                    self.vision_adapter,
                    "detect_faces",
                )
            ):

                results = (
                    self.vision_adapter.detect_faces(
                        frame
                    )
                )

                for face in results or []:

                    normalized = (
                        self._normalize_detection(
                            face,
                            default_label="face",
                        )
                    )

                    faces.append(
                        normalized
                    )

        except Exception as error:

            logger.error(
                "Face detection failed: %s",
                error,
            )

        return faces

    # ============================================================
    # DEPTH ESTIMATION
    # ============================================================

    def estimate_depth(
        self,
        frame: Any,
    ) -> Optional[Any]:
        """
        Estimate scene depth.

        Can work with:
        - Depth cameras
        - Stereo cameras
        - AI depth estimation
        """

        try:

            if (
                self.vision_adapter is not None
                and hasattr(
                    self.vision_adapter,
                    "estimate_depth",
                )
            ):

                return (
                    self.vision_adapter.estimate_depth(
                        frame
                    )
                )

        except Exception as error:

            logger.error(
                "Depth estimation failed: %s",
                error,
            )

        return None

    # ============================================================
    # DISTANCE ESTIMATION
    # ============================================================

    def estimate_object_distance(
        self,
        detection: Dict[str, Any],
        depth_data: Optional[Any] = None,
    ) -> Optional[float]:
        """
        Estimate distance to an object.

        Priority:
        1. Detection distance
        2. Depth map adapter
        3. Approximation from bounding box size
        """

        if (
            "distance"
            in detection
            and detection["distance"]
            is not None
        ):

            try:
                return float(
                    detection["distance"]
                )
            except (
                ValueError,
                TypeError,
            ):
                pass

        try:

            if (
                self.vision_adapter is not None
                and hasattr(
                    self.vision_adapter,
                    "estimate_object_distance",
                )
            ):

                return float(
                    self.vision_adapter
                    .estimate_object_distance(
                        detection,
                        depth_data,
                    )
                )

        except Exception as error:

            logger.debug(
                "Adapter distance estimation failed: %s",
                error,
            )

        bbox = detection.get(
            "bbox"
        )

        if (
            bbox
            and len(bbox) == 4
        ):

            _, _, width, height = bbox

            area = (
                max(width, 1)
                * max(height, 1)
            )

            estimated_distance = (
                10000 / math.sqrt(area)
            )

            return round(
                estimated_distance,
                2,
            )

        return None

    # ============================================================
    # OBSTACLE DETECTION
    # ============================================================

    def detect_obstacles(
        self,
        objects: Optional[
            List[Dict[str, Any]]
        ] = None,
        depth_data: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect nearby obstacles.
        """

        if objects is None:

            objects = (
                self.detected_objects
            )

        obstacles = []

        for obj in objects:

            distance = (
                self.estimate_object_distance(
                    obj,
                    depth_data,
                )
            )

            if distance is None:
                continue

            if (
                distance
                <= self.obstacle_distance_threshold
            ):

                obstacle = dict(obj)

                obstacle[
                    "distance"
                ] = distance

                obstacles.append(
                    obstacle
                )

        return sorted(
            obstacles,
            key=lambda item: item.get(
                "distance",
                float("inf"),
            ),
        )

    # ============================================================
    # TARGET SELECTION
    # ============================================================

    def select_target(
        self,
        target: Dict[str, Any],
    ) -> bool:
        """
        Select an object/person as the current robot target.
        """

        if not target:
            return False

        with self._lock:

            self.current_target = dict(
                target
            )

        self._emit_event(
            "target_selected",
            target=self.current_target,
        )

        return True

    def select_target_by_label(
        self,
        label: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Find and select the highest-confidence
        object with a given label.
        """

        label = (
            str(label)
            .strip()
            .lower()
        )

        candidates = [
            obj
            for obj in self.detected_objects
            if obj[
                "label"
            ].lower()
            == label
        ]

        if not candidates:
            return None

        target = max(
            candidates,
            key=lambda item: item.get(
                "confidence",
                0,
            ),
        )

        self.select_target(
            target
        )

        return target

    def clear_target(
        self,
    ) -> None:
        """Clear the current target."""

        with self._lock:

            self.current_target = None

        self._emit_event(
            "target_cleared"
        )

    def get_target(
        self,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """Return current robot target."""

        if self.current_target is None:
            return None

        return dict(
            self.current_target
        )

    # ============================================================
    # TRACKING
    # ============================================================

    def track_object(
        self,
        object_id: str,
        detection: Dict[str, Any],
    ) -> bool:
        """
        Add or update tracked object.
        """

        if not object_id:
            return False

        object_id = str(
            object_id
        )

        tracking_data = {
            "object_id": object_id,
            "detection": dict(
                detection
            ),
            "last_seen": (
                datetime.now()
                .isoformat()
            ),
        }

        with self._lock:

            self.tracked_objects[
                object_id
            ] = tracking_data

        return True

    def stop_tracking(
        self,
        object_id: str,
    ) -> bool:
        """Stop tracking an object."""

        object_id = str(
            object_id
        )

        with self._lock:

            if (
                object_id
                not in self.tracked_objects
            ):

                return False

            del self.tracked_objects[
                object_id
            ]

        return True

    def get_tracked_objects(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """Return tracked objects."""

        with self._lock:

            return {
                key: dict(value)
                for key, value
                in self.tracked_objects.items()
            }

    def _update_tracking(
        self,
        objects: List[
            Dict[str, Any]
        ],
    ) -> None:
        """
        Update tracked object information.
        """

        for obj in objects:

            object_id = obj.get(
                "object_id"
            )

            if object_id is None:
                continue

            object_id = str(
                object_id
            )

            if (
                object_id
                in self.tracked_objects
            ):

                self.track_object(
                    object_id,
                    obj,
                )

    # ============================================================
    # SPATIAL INFORMATION
    # ============================================================

    def get_object_position(
        self,
        detection: Dict[str, Any],
    ) -> Dict[str, Optional[float]]:
        """
        Estimate object's relative position.

        Returns:
        - x_center
        - y_center
        - relative_direction
        """

        bbox = detection.get(
            "bbox"
        )

        frame_width = detection.get(
            "frame_width"
        )

        frame_height = detection.get(
            "frame_height"
        )

        result = {
            "x_center": None,
            "y_center": None,
            "direction": "unknown",
        }

        if (
            not bbox
            or len(bbox) != 4
        ):

            return result

        x, y, width, height = bbox

        center_x = (
            x + width / 2
        )

        center_y = (
            y + height / 2
        )

        result[
            "x_center"
        ] = center_x

        result[
            "y_center"
        ] = center_y

        if frame_width:

            relative_x = (
                center_x
                / frame_width
            )

            if relative_x < 0.33:

                result[
                    "direction"
                ] = "left"

            elif relative_x > 0.66:

                result[
                    "direction"
                ] = "right"

            else:

                result[
                    "direction"
                ] = "center"

        return result

    # ============================================================
    # FOLLOWING SUPPORT
    # ============================================================

    def get_follow_command(
        self,
        target: Optional[
            Dict[str, Any]
        ] = None,
        desired_distance: float = 150.0,
    ) -> Dict[str, Any]:
        """
        Generate high-level follow guidance.

        This does not move the robot directly.
        It returns movement suggestions for
        robot_controller.py or navigation.py.
        """

        if target is None:

            target = self.current_target

        if target is None:

            return {
                "success": False,
                "action": "stop",
                "reason": "No target.",
            }

        position = (
            self.get_object_position(
                target
            )
        )

        distance = (
            self.estimate_object_distance(
                target
            )
        )

        direction = position.get(
            "direction",
            "unknown",
        )

        command = {
            "success": True,
            "target": target.get(
                "label",
                "unknown",
            ),
            "direction": direction,
            "distance": distance,
            "action": "hold",
        }

        if direction == "left":

            command[
                "action"
            ] = "turn_left"

        elif direction == "right":

            command[
                "action"
            ] = "turn_right"

        elif distance is not None:

            if distance > desired_distance:

                command[
                    "action"
                ] = "move_forward"

            elif distance < (
                desired_distance * 0.7
            ):

                command[
                    "action"
                ] = "move_backward"

            else:

                command[
                    "action"
                ] = "hold"

        return command

    # ============================================================
    # HISTORY
    # ============================================================

    def get_analysis_history(
        self,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return recent scene analysis history."""

        history = list(
            self.analysis_history
        )

        return history[
            -max(
                1,
                int(limit),
            ):
        ]

    def clear_history(
        self,
    ) -> None:
        """Clear vision analysis history."""

        with self._lock:

            self.analysis_history.clear()

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
        """Register a vision event handler."""

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
        """Remove a vision event handler."""

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
        """Emit robot vision event."""

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
                    "Vision event handler error: %s",
                    error,
                )

    # ============================================================
    # HELPERS
    # ============================================================

    def _normalize_detection(
        self,
        detection: Any,
        default_label: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Normalize different model outputs
        into a common RENIX format.
        """

        if isinstance(
            detection,
            dict,
        ):

            return {
                "object_id": detection.get(
                    "object_id"
                ),
                "label": detection.get(
                    "label",
                    detection.get(
                        "class",
                        default_label,
                    ),
                ),
                "confidence": float(
                    detection.get(
                        "confidence",
                        detection.get(
                            "score",
                            1.0,
                        ),
                    )
                ),
                "bbox": detection.get(
                    "bbox"
                ),
                "distance": detection.get(
                    "distance"
                ),
                "frame_width": detection.get(
                    "frame_width"
                ),
                "frame_height": detection.get(
                    "frame_height"
                ),
                "metadata": detection.get(
                    "metadata",
                    {},
                ),
            }

        return {
            "object_id": None,
            "label": default_label,
            "confidence": 1.0,
            "bbox": None,
            "distance": None,
            "frame_width": None,
            "frame_height": None,
            "metadata": {
                "raw": str(
                    detection
                ),
            },
        }

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """Return robot vision status."""

        return {
            "connected": self.connected,
            "running": self.running,
            "objects_detected": len(
                self.detected_objects
            ),
            "people_detected": len(
                self.detected_people
            ),
            "faces_detected": len(
                self.detected_faces
            ),
            "tracked_objects": len(
                self.tracked_objects
            ),
            "current_target": (
                self.get_target()
            ),
            "history_size": len(
                self.analysis_history
            ),
        }

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """Safely shutdown RobotVision."""

        logger.info(
            "Shutting down RobotVision..."
        )

        self.stop()

        self.disconnect()

        with self._lock:

            self.detected_objects.clear()
            self.detected_people.clear()
            self.detected_faces.clear()
            self.tracked_objects.clear()
            self.analysis_history.clear()
            self.current_target = None
            self.event_handlers.clear()

        logger.info(
            "RobotVision shutdown complete."
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


    class DemoVisionAdapter:

        def connect(self):

            print(
                "Camera connected."
            )

            return True

        def disconnect(self):

            print(
                "Camera disconnected."
            )

            return True

        def get_frame(self):

            return "demo_frame"

        def detect_objects(
            self,
            frame,
        ):

            return [
                {
                    "object_id": "1",
                    "label": "person",
                    "confidence": 0.95,
                    "bbox": (
                        300,
                        100,
                        200,
                        400,
                    ),
                    "distance": 180,
                    "frame_width": 1280,
                    "frame_height": 720,
                },
                {
                    "object_id": "2",
                    "label": "chair",
                    "confidence": 0.88,
                    "bbox": (
                        100,
                        300,
                        150,
                        200,
                    ),
                    "distance": 80,
                    "frame_width": 1280,
                    "frame_height": 720,
                },
            ]

        def detect_faces(
            self,
            frame,
        ):

            return [
                {
                    "label": "face",
                    "confidence": 0.99,
                    "bbox": (
                        350,
                        120,
                        100,
                        100,
                    ),
                }
            ]


    vision = RobotVision(
        vision_adapter=DemoVisionAdapter(),
        config={
            "confidence_threshold": 0.5,
            "obstacle_distance_threshold": 100,
        },
    )

    vision.connect()

    result = vision.analyze_scene()

    print("\nScene Analysis:")
    print(result)

    target = (
        vision.select_target_by_label(
            "person"
        )
    )

    print("\nSelected Target:")
    print(target)

    print("\nFollow Command:")
    print(
        vision.get_follow_command(
            desired_distance=150
        )
    )

    print("\nRobot Vision Status:")
    print(
        vision.get_status()
    )

    vision.shutdown()


