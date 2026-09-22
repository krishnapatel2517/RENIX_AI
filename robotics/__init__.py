"""
RENIX Robotics Module
=====================

Provides robotics capabilities for RENIX.

Features:

* Robot management
* Robot control
* Servo control
* Sensor management
* Robot vision
* Navigation
* Object manipulation
* Robotics safety

The robotics module is designed to support different robot
implementations, including:

* Arduino-based robots
* Raspberry Pi robots
* ESP32 robots
* Robotic arms
* Mobile robots
* Smart vehicles
* Custom hardware platforms
  """

from .robot_manager import RobotManager
from .robot_controller import RobotController
from .servo_controller import ServoController
from .sensor_manager import SensorManager
from .robot_vision import RobotVision
from .navigation import Navigation
from .object_manipulation import ObjectManipulation
from .safety import RoboticsSafety

__all__ = [
"RobotManager",
"RobotController",
"ServoController",
"SensorManager",
"RobotVision",
"Navigation",
"ObjectManipulation",
"RoboticsSafety",
]

__version__ = "1.0.0"
__author__ = "RENIX"



