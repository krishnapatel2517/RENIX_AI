"""
RENIX Computer Control System
=============================

Provides the core interface for controlling and interacting
with the user's computer.

Modules:
    computer_controller
    mouse
    keyboard
    screen
    windows
    applications
    clipboard
    screenshots
    screen_reader
    desktop
    system_settings
    volume
    brightness
    display
    process_manager
"""

from .computer_controller import ComputerController
from .mouse import MouseController
from .keyboard import KeyboardController
from .screen import ScreenController
from .windows import WindowManager
from .applications import ApplicationManager
from .clipboard import ClipboardManager
from .screenshots import ScreenshotManager
from .screen_reader import ScreenReader
from .desktop import DesktopManager
from .system_settings import SystemSettings
from .volume import VolumeController
from .brightness import BrightnessController
from .display import DisplayController
from .process_manager import ProcessManager


__all__ = [
    "ComputerController",
    "MouseController",
    "KeyboardController",
    "ScreenController",
    "WindowManager",
    "ApplicationManager",
    "ClipboardManager",
    "ScreenshotManager",
    "ScreenReader",
    "DesktopManager",
    "SystemSettings",
    "VolumeController",
    "BrightnessController",
    "DisplayController",
    "ProcessManager",
]


__version__ = "1.0.0"
__author__ = "RENIX"


