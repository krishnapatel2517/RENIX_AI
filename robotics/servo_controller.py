"""
RENIX Servo Controller
======================

Controls single or multiple servo motors.

Features:
- Register multiple servos
- Set servo angle
- Smooth movement
- Speed-controlled movement
- Center servo
- Attach / detach servo
- Angle limits
- Servo presets / poses
- Robotic arm joint support
- Hardware adapter support
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional


logger = logging.getLogger("RENIX.Robotics.ServoController")


class ServoController:
    """
    High-level controller for servo motors.

    Hardware-independent design.

    The adapter may implement methods such as:

        connect()
        disconnect()
        attach_servo(servo_id, pin)
        detach_servo(servo_id)
        set_servo_angle(servo_id, angle)
    """

    def __init__(
        self,
        adapter: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.adapter = adapter
        self.config = config or {}

        self.servos: Dict[str, Dict[str, Any]] = {}
        self.presets: Dict[str, Dict[str, float]] = {}

        self.default_min_angle = int(
            self.config.get("min_angle", 0)
        )

        self.default_max_angle = int(
            self.config.get("max_angle", 180)
        )

        self.default_angle = int(
            self.config.get("default_angle", 90)
        )

        self.connected = False

        self._lock = threading.RLock()

        logger.info("ServoController initialized.")

    # ============================================================
    # CONNECTION
    # ============================================================

    def connect(self) -> bool:
        """Connect to the servo hardware."""

        if self.connected:
            return True

        try:
            if self.adapter and hasattr(self.adapter, "connect"):
                result = self.adapter.connect()

                if result is False:
                    return False

            self.connected = True

            logger.info("Servo hardware connected.")

            return True

        except Exception as error:

            logger.error(
                "Failed to connect servo controller: %s",
                error,
            )

            return False

    def disconnect(self) -> bool:
        """Disconnect servo hardware."""

        try:

            if self.adapter and hasattr(
                self.adapter,
                "disconnect",
            ):
                result = self.adapter.disconnect()

                if result is False:
                    return False

            self.connected = False

            logger.info("Servo hardware disconnected.")

            return True

        except Exception as error:

            logger.error(
                "Failed to disconnect servo controller: %s",
                error,
            )

            return False

    # ============================================================
    # SERVO REGISTRATION
    # ============================================================

    def register_servo(
        self,
        servo_id: str,
        pin: Optional[int] = None,
        min_angle: Optional[int] = None,
        max_angle: Optional[int] = None,
        default_angle: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Register a servo motor.

        Example:

            register_servo(
                servo_id="shoulder",
                pin=9,
                min_angle=10,
                max_angle=170,
            )
        """

        servo_id = str(servo_id).strip().lower()

        min_angle = (
            self.default_min_angle
            if min_angle is None
            else int(min_angle)
        )

        max_angle = (
            self.default_max_angle
            if max_angle is None
            else int(max_angle)
        )

        default_angle = (
            self.default_angle
            if default_angle is None
            else int(default_angle)
        )

        if min_angle >= max_angle:

            return {
                "success": False,
                "error": (
                    "min_angle must be smaller than max_angle."
                ),
            }

        default_angle = max(
            min_angle,
            min(default_angle, max_angle),
        )

        with self._lock:

            self.servos[servo_id] = {
                "servo_id": servo_id,
                "pin": pin,
                "min_angle": min_angle,
                "max_angle": max_angle,
                "current_angle": default_angle,
                "default_angle": default_angle,
                "attached": False,
            }

        logger.info(
            "Servo registered: %s",
            servo_id,
        )

        return {
            "success": True,
            "servo": self.get_servo(servo_id),
        }

    def unregister_servo(
        self,
        servo_id: str,
    ) -> bool:
        """Remove a servo."""

        servo_id = str(servo_id).strip().lower()

        self.detach_servo(servo_id)

        with self._lock:

            if servo_id not in self.servos:
                return False

            del self.servos[servo_id]

        return True

    # ============================================================
    # ATTACH / DETACH
    # ============================================================

    def attach_servo(
        self,
        servo_id: str,
    ) -> Dict[str, Any]:
        """Attach a servo motor."""

        servo = self._get_servo_reference(servo_id)

        if servo is None:

            return {
                "success": False,
                "error": "Servo not found.",
            }

        try:

            if (
                self.adapter
                and hasattr(
                    self.adapter,
                    "attach_servo",
                )
            ):

                result = self.adapter.attach_servo(
                    servo["servo_id"],
                    servo["pin"],
                )

                if result is False:

                    return {
                        "success": False,
                        "error": (
                            "Hardware refused servo attachment."
                        ),
                    }

            servo["attached"] = True

            return {
                "success": True,
                "servo_id": servo["servo_id"],
                "attached": True,
            }

        except Exception as error:

            logger.error(
                "Failed to attach servo %s: %s",
                servo_id,
                error,
            )

            return {
                "success": False,
                "error": str(error),
            }

    def detach_servo(
        self,
        servo_id: str,
    ) -> Dict[str, Any]:
        """Detach a servo motor."""

        servo = self._get_servo_reference(servo_id)

        if servo is None:

            return {
                "success": False,
                "error": "Servo not found.",
            }

        try:

            if (
                self.adapter
                and hasattr(
                    self.adapter,
                    "detach_servo",
                )
            ):

                result = self.adapter.detach_servo(
                    servo["servo_id"]
                )

                if result is False:

                    return {
                        "success": False,
                    }

            servo["attached"] = False

            return {
                "success": True,
                "servo_id": servo["servo_id"],
                "attached": False,
            }

        except Exception as error:

            return {
                "success": False,
                "error": str(error),
            }

    # ============================================================
    # ANGLE CONTROL
    # ============================================================

    def set_angle(
        self,
        servo_id: str,
        angle: float,
    ) -> Dict[str, Any]:
        """
        Set servo angle immediately.
        """

        servo = self._get_servo_reference(servo_id)

        if servo is None:

            return {
                "success": False,
                "error": "Servo not found.",
            }

        angle = self._normalize_angle(
            servo,
            angle,
        )

        try:

            if (
                self.adapter
                and hasattr(
                    self.adapter,
                    "set_servo_angle",
                )
            ):

                result = self.adapter.set_servo_angle(
                    servo["servo_id"],
                    angle,
                )

                if result is False:

                    return {
                        "success": False,
                        "error": (
                            "Hardware rejected servo command."
                        ),
                    }

            elif (
                self.adapter
                and hasattr(
                    self.adapter,
                    "send",
                )
            ):

                self.adapter.send(
                    {
                        "action": "servo_angle",
                        "servo_id": servo["servo_id"],
                        "angle": angle,
                    }
                )

            servo["current_angle"] = angle

            return {
                "success": True,
                "servo_id": servo["servo_id"],
                "angle": angle,
            }

        except Exception as error:

            logger.error(
                "Failed to set servo angle: %s",
                error,
            )

            return {
                "success": False,
                "error": str(error),
            }

    def get_angle(
        self,
        servo_id: str,
    ) -> Optional[float]:
        """Get current servo angle."""

        servo = self._get_servo_reference(servo_id)

        if servo is None:
            return None

        return servo["current_angle"]

    # ============================================================
    # SMOOTH MOVEMENT
    # ============================================================

    def move_smooth(
        self,
        servo_id: str,
        target_angle: float,
        speed: float = 90.0,
        step_size: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Move servo smoothly.

        Args:
            target_angle:
                Final angle.

            speed:
                Approximate degrees per second.

            step_size:
                Angle increment per movement step.
        """

        servo = self._get_servo_reference(servo_id)

        if servo is None:

            return {
                "success": False,
                "error": "Servo not found.",
            }

        current = float(
            servo["current_angle"]
        )

        target = float(
            self._normalize_angle(
                servo,
                target_angle,
            )
        )

        if current == target:

            return {
                "success": True,
                "servo_id": servo_id,
                "angle": target,
            }

        step_size = max(
            0.1,
            float(step_size),
        )

        speed = max(
            1.0,
            float(speed),
        )

        direction = (
            1
            if target > current
            else -1
        )

        delay = (
            step_size / speed
        )

        angle = current

        try:

            while (
                (direction > 0 and angle < target)
                or
                (direction < 0 and angle > target)
            ):

                angle += (
                    direction * step_size
                )

                if direction > 0:
                    angle = min(
                        angle,
                        target,
                    )
                else:
                    angle = max(
                        angle,
                        target,
                    )

                result = self.set_angle(
                    servo_id,
                    angle,
                )

                if not result["success"]:

                    return result

                time.sleep(delay)

            return {
                "success": True,
                "servo_id": servo_id,
                "angle": target,
            }

        except Exception as error:

            logger.error(
                "Smooth movement failed: %s",
                error,
            )

            return {
                "success": False,
                "error": str(error),
            }

    # ============================================================
    # CENTER / RESET
    # ============================================================

    def center_servo(
        self,
        servo_id: str,
    ) -> Dict[str, Any]:
        """Move servo to its center position."""

        servo = self._get_servo_reference(servo_id)

        if servo is None:

            return {
                "success": False,
                "error": "Servo not found.",
            }

        center = (
            servo["min_angle"]
            + servo["max_angle"]
        ) / 2

        return self.set_angle(
            servo_id,
            center,
        )

    def reset_servo(
        self,
        servo_id: str,
    ) -> Dict[str, Any]:
        """Reset servo to default angle."""

        servo = self._get_servo_reference(servo_id)

        if servo is None:

            return {
                "success": False,
                "error": "Servo not found.",
            }

        return self.set_angle(
            servo_id,
            servo["default_angle"],
        )

    def reset_all(
        self,
    ) -> Dict[str, Any]:
        """Reset all servos."""

        results = {}

        for servo_id in list(
            self.servos.keys()
        ):

            results[servo_id] = (
                self.reset_servo(
                    servo_id
                )
            )

        return {
            "success": all(
                result.get(
                    "success",
                    False,
                )
                for result in results.values()
            ),
            "results": results,
        }

    # ============================================================
    # MULTIPLE SERVOS
    # ============================================================

    def set_multiple_angles(
        self,
        angles: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Set multiple servo angles.

        Example:

            set_multiple_angles({
                "shoulder": 90,
                "elbow": 120,
                "wrist": 45
            })
        """

        results = {}

        for servo_id, angle in angles.items():

            results[servo_id] = (
                self.set_angle(
                    servo_id,
                    angle,
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
    # PRESETS / ROBOT POSES
    # ============================================================

    def save_preset(
        self,
        name: str,
        angles: Dict[str, float],
    ) -> bool:
        """
        Save a robot pose.

        Example:

            save_preset(
                "wave",
                {
                    "shoulder": 90,
                    "elbow": 45,
                    "wrist": 120
                }
            )
        """

        name = str(name).strip().lower()

        validated_angles = {}

        for servo_id, angle in angles.items():

            servo = self._get_servo_reference(
                servo_id
            )

            if servo is None:
                continue

            validated_angles[
                servo_id
            ] = self._normalize_angle(
                servo,
                angle,
            )

        if not validated_angles:
            return False

        with self._lock:

            self.presets[name] = (
                validated_angles
            )

        logger.info(
            "Servo preset saved: %s",
            name,
        )

        return True

    def load_preset(
        self,
        name: str,
        smooth: bool = False,
        speed: float = 90.0,
    ) -> Dict[str, Any]:
        """
        Load a saved servo preset.
        """

        name = str(name).strip().lower()

        preset = self.presets.get(name)

        if preset is None:

            return {
                "success": False,
                "error": (
                    f"Preset '{name}' not found."
                ),
            }

        results = {}

        for servo_id, angle in preset.items():

            if smooth:

                results[servo_id] = (
                    self.move_smooth(
                        servo_id,
                        angle,
                        speed,
                    )
                )

            else:

                results[servo_id] = (
                    self.set_angle(
                        servo_id,
                        angle,
                    )
                )

        return {
            "success": all(
                result.get(
                    "success",
                    False,
                )
                for result in results.values()
            ),
            "preset": name,
            "results": results,
        }

    def delete_preset(
        self,
        name: str,
    ) -> bool:
        """Delete a servo preset."""

        name = str(name).strip().lower()

        with self._lock:

            if name not in self.presets:
                return False

            del self.presets[name]

        return True

    def list_presets(
        self,
    ) -> List[str]:
        """Return available preset names."""

        return sorted(
            self.presets.keys()
        )

    # ============================================================
    # INFORMATION
    # ============================================================

    def get_servo(
        self,
        servo_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return servo information."""

        servo = self._get_servo_reference(
            servo_id
        )

        if servo is None:
            return None

        return dict(servo)

    def get_all_servos(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """Return all registered servos."""

        with self._lock:

            return {
                servo_id: dict(data)
                for servo_id, data
                in self.servos.items()
            }

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """Return complete servo controller status."""

        return {
            "connected": self.connected,
            "servo_count": len(
                self.servos
            ),
            "servos": self.get_all_servos(),
            "presets": self.list_presets(),
        }

    # ============================================================
    # HELPERS
    # ============================================================

    def _get_servo_reference(
        self,
        servo_id: str,
    ) -> Optional[Dict[str, Any]]:

        servo_id = str(
            servo_id
        ).strip().lower()

        with self._lock:

            return self.servos.get(
                servo_id
            )

    def _normalize_angle(
        self,
        servo: Dict[str, Any],
        angle: float,
    ) -> float:
        """Keep angle inside servo limits."""

        angle = float(angle)

        return max(
            servo["min_angle"],
            min(
                angle,
                servo["max_angle"],
            ),
        )

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """Safely shutdown servo controller."""

        logger.info(
            "Shutting down ServoController..."
        )

        try:
            self.reset_all()
        except Exception:
            pass

        self.disconnect()

        logger.info(
            "ServoController shutdown complete."
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

    class DemoServoAdapter:

        def connect(self):

            print(
                "Servo hardware connected."
            )

            return True

        def disconnect(self):

            print(
                "Servo hardware disconnected."
            )

            return True

        def attach_servo(
            self,
            servo_id,
            pin,
        ):

            print(
                f"Attached {servo_id} "
                f"to pin {pin}"
            )

            return True

        def set_servo_angle(
            self,
            servo_id,
            angle,
        ):

            print(
                f"{servo_id} -> {angle}°"
            )

            return True


    adapter = DemoServoAdapter()

    controller = ServoController(
        adapter=adapter
    )

    controller.connect()

    controller.register_servo(
        "shoulder",
        pin=9,
        min_angle=10,
        max_angle=170,
        default_angle=90,
    )

    controller.register_servo(
        "elbow",
        pin=10,
        min_angle=20,
        max_angle=160,
        default_angle=90,
    )

    controller.attach_servo(
        "shoulder"
    )

    controller.attach_servo(
        "elbow"
    )

    controller.set_angle(
        "shoulder",
        120,
    )

    controller.move_smooth(
        "elbow",
        130,
        speed=120,
    )

    controller.save_preset(
        "ready",
        {
            "shoulder": 90,
            "elbow": 90,
        },
    )

    controller.load_preset(
        "ready"
    )

    print(
        controller.get_status()
    )

    controller.shutdown()


