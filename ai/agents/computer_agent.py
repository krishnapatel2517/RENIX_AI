"""
RENIX AI - Computer Agent

Controls and interacts with the computer through RENIX's computer layer.

Responsibilities:
- Mouse control
- Keyboard control
- Application launching
- Window management
- Screenshots
- Clipboard operations
- Desktop interaction
- Screen information
- Basic system controls

This agent communicates with the existing `computer/` package.
It does not create additional files.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base_agent import (
    AgentCapability,
    AgentPriority,
    AgentResult,
    AgentResultStatus,
    AgentTask,
    BaseAgent,
)


class ComputerAgent(BaseAgent):
    """
    RENIX computer-control agent.

    This agent provides a unified interface between the AI orchestration
    layer and the computer subsystem.
    """

    agent_name = "computer_agent"

    agent_description = (
        "Controls the computer, applications, windows, mouse, keyboard, "
        "clipboard, screenshots and desktop."
    )

    agent_version = "1.0.0"

    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        priority: AgentPriority = AgentPriority.HIGH,
        logger: Optional[logging.Logger] = None,
        event_callback=None,
        confirmation_callback=None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            priority=priority,
            logger=logger,
            event_callback=event_callback,
            confirmation_callback=confirmation_callback,
            context=context,
        )

        self.system = platform.system().lower()

        self.computer_controller = None
        self.mouse_controller = None
        self.keyboard_controller = None
        self.screen_controller = None
        self.window_controller = None
        self.application_controller = None
        self.clipboard_controller = None
        self.screenshot_controller = None
        self.desktop_controller = None
        self.volume_controller = None
        self.brightness_controller = None
        self.display_controller = None
        self.process_controller = None

        self._computer_modules_loaded = False

    # ========================================================================
    # CAPABILITIES
    # ========================================================================

    def _register_default_capabilities(self) -> None:
        """Register computer-agent capabilities."""

        super()._register_default_capabilities()

        capabilities = [
            AgentCapability(
                name="mouse",
                description=(
                    "Control mouse movement, clicks and scrolling."
                ),
            ),
            AgentCapability(
                name="keyboard",
                description=(
                    "Type text and send keyboard commands."
                ),
            ),
            AgentCapability(
                name="applications",
                description=(
                    "Open, close and manage applications."
                ),
                requires_confirmation=True,
            ),
            AgentCapability(
                name="windows",
                description=(
                    "Move, resize, minimize, maximize and close windows."
                ),
                requires_confirmation=True,
            ),
            AgentCapability(
                name="screen",
                description=(
                    "Read screen information and interact with displays."
                ),
            ),
            AgentCapability(
                name="screenshot",
                description=(
                    "Capture screenshots."
                ),
            ),
            AgentCapability(
                name="clipboard",
                description=(
                    "Read and modify clipboard contents."
                ),
            ),
            AgentCapability(
                name="desktop",
                description=(
                    "Interact with the desktop."
                ),
            ),
            AgentCapability(
                name="volume",
                description=(
                    "Control system volume."
                ),
            ),
            AgentCapability(
                name="brightness",
                description=(
                    "Control display brightness when supported."
                ),
            ),
            AgentCapability(
                name="display",
                description=(
                    "Read and control display configuration."
                ),
            ),
            AgentCapability(
                name="processes",
                description=(
                    "Inspect and manage system processes."
                ),
                requires_confirmation=True,
            ),
            AgentCapability(
                name="system_command",
                description=(
                    "Execute a system command."
                ),
                dangerous=True,
                requires_confirmation=True,
            ),
        ]

        for capability in capabilities:
            self.register_capability(capability)

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    async def on_initialize(self) -> bool:
        """
        Load the RENIX computer subsystem.

        Missing optional modules do not prevent RENIX from starting.
        """

        self._load_computer_modules()

        return True

    def _load_computer_modules(self) -> None:
        """Load available modules from the existing computer package."""

        if self._computer_modules_loaded:
            return

        self._computer_modules_loaded = True

        # --------------------------------------------------------------------
        # COMPUTER CONTROLLER
        # --------------------------------------------------------------------

        try:
            from computer.computer_controller import (
                ComputerController,
            )

            self.computer_controller = ComputerController()

        except Exception as exc:
            self.logger.debug(
                "ComputerController unavailable: %s",
                exc,
            )

        # --------------------------------------------------------------------
        # MOUSE
        # --------------------------------------------------------------------

        try:
            from computer.mouse import MouseController

            self.mouse_controller = MouseController()

        except Exception as exc:
            self.logger.debug(
                "MouseController unavailable: %s",
                exc,
            )

        # --------------------------------------------------------------------
        # KEYBOARD
        # --------------------------------------------------------------------

        try:
            from computer.keyboard import KeyboardController

            self.keyboard_controller = KeyboardController()

        except Exception as exc:
            self.logger.debug(
                "KeyboardController unavailable: %s",
                exc,
            )

        # --------------------------------------------------------------------
        # SCREEN
        # --------------------------------------------------------------------

        try:
            from computer.screen import ScreenController

            self.screen_controller = ScreenController()

        except Exception as exc:
            self.logger.debug(
                "ScreenController unavailable: %s",
                exc,
            )

        # --------------------------------------------------------------------
        # WINDOWS
        # --------------------------------------------------------------------

        try:
            from computer.windows import WindowController

            self.window_controller = WindowController()

        except Exception as exc:
            self.logger.debug(
                "WindowController unavailable: %s",
                exc,
            )

        # --------------------------------------------------------------------
        # APPLICATIONS
        # --------------------------------------------------------------------

        try:
            from computer.applications import ApplicationController

            self.application_controller = ApplicationController()

        except Exception as exc:
            self.logger.debug(
                "ApplicationController unavailable: %s",
                exc,
            )

        # --------------------------------------------------------------------
        # CLIPBOARD
        # --------------------------------------------------------------------

        try:
            from computer.clipboard import ClipboardController

            self.clipboard_controller = ClipboardController()

        except Exception as exc:
            self.logger.debug(
                "ClipboardController unavailable: %s",
                exc,
            )

        # --------------------------------------------------------------------
        # SCREENSHOTS
        # --------------------------------------------------------------------

        try:
            from computer.screenshots import ScreenshotController

            self.screenshot_controller = ScreenshotController()

        except Exception as exc:
            self.logger.debug(
                "ScreenshotController unavailable: %s",
                exc,
            )

        # --------------------------------------------------------------------
        # DESKTOP
        # --------------------------------------------------------------------

        try:
            from computer.desktop import DesktopController

            self.desktop_controller = DesktopController()

        except Exception as exc:
            self.logger.debug(
                "DesktopController unavailable: %s",
                exc,
            )

        # --------------------------------------------------------------------
        # VOLUME
        # --------------------------------------------------------------------

        try:
            from computer.volume import VolumeController

            self.volume_controller = VolumeController()

        except Exception as exc:
            self.logger.debug(
                "VolumeController unavailable: %s",
                exc,
            )

        # --------------------------------------------------------------------
        # BRIGHTNESS
        # --------------------------------------------------------------------

        try:
            from computer.brightness import BrightnessController

            self.brightness_controller = BrightnessController()

        except Exception as exc:
            self.logger.debug(
                "BrightnessController unavailable: %s",
                exc,
            )

        # --------------------------------------------------------------------
        # DISPLAY
        # --------------------------------------------------------------------

        try:
            from computer.display import DisplayController

            self.display_controller = DisplayController()

        except Exception as exc:
            self.logger.debug(
                "DisplayController unavailable: %s",
                exc,
            )

        # --------------------------------------------------------------------
        # PROCESSES
        # --------------------------------------------------------------------

        try:
            from computer.process_manager import ProcessManager

            self.process_controller = ProcessManager()

        except Exception as exc:
            self.logger.debug(
                "ProcessManager unavailable: %s",
                exc,
            )

    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================

    async def execute(
        self,
        task: AgentTask,
    ) -> Any:
        """
        Execute a computer task.

        Supported task actions include:

        mouse_move
        mouse_click
        mouse_double_click
        mouse_scroll
        keyboard_type
        keyboard_press
        hotkey
        screenshot
        clipboard_get
        clipboard_set
        open_application
        close_application
        window_minimize
        window_maximize
        window_close
        window_focus
        volume
        brightness
        screen_info
        desktop
        process_list
        process_kill
        system_command
        launch_url
        open_file
        """

        action = self._resolve_action(task)

        handlers = {
            "mouse_move": self.mouse_move,
            "mouse_click": self.mouse_click,
            "mouse_double_click": self.mouse_double_click,
            "mouse_scroll": self.mouse_scroll,
            "keyboard_type": self.keyboard_type,
            "keyboard_press": self.keyboard_press,
            "hotkey": self.hotkey,
            "screenshot": self.screenshot,
            "take_screenshot": self.screenshot,
            "clipboard_get": self.clipboard_get,
            "clipboard_set": self.clipboard_set,
            "open_application": self.open_application,
            "close_application": self.close_application,
            "window_minimize": self.window_minimize,
            "window_maximize": self.window_maximize,
            "window_close": self.window_close,
            "window_focus": self.window_focus,
            "volume": self.set_volume,
            "brightness": self.set_brightness,
            "screen_info": self.get_screen_info,
            "desktop": self.show_desktop,
            "process_list": self.list_processes,
            "process_kill": self.kill_process,
            "system_command": self.execute_system_command,
            "launch_url": self.launch_url,
            "open_file": self.open_file,
        }

        handler = handlers.get(action)

        if handler is None:
            return self.failure_result(
                task,
                f"Unknown computer action: {action}",
                started_at=task.created_at,
            )

        try:
            parameters = dict(task.parameters)

            parameters.pop("action", None)
            parameters.pop("command", None)
            parameters.pop("capability", None)

            result = handler(**parameters)

            if asyncio.iscoroutine(result):
                result = await result

            return result

        except Exception as exc:
            self.logger.exception(
                "Computer action failed: %s",
                action,
            )

            return self.failure_result(
                task,
                str(exc),
                message=f"Computer action '{action}' failed.",
                started_at=task.created_at,
            )

    # ========================================================================
    # ACTION RESOLUTION
    # ========================================================================

    def _resolve_action(
        self,
        task: AgentTask,
    ) -> str:
        """Determine the requested computer action."""

        action = task.parameters.get("action")

        if action:
            return str(action).strip().lower()

        instruction = task.instruction.lower().strip()

        if instruction.startswith("mouse move"):
            return "mouse_move"

        if instruction.startswith("click"):
            return "mouse_click"

        if instruction.startswith("double click"):
            return "mouse_double_click"

        if instruction.startswith("type"):
            return "keyboard_type"

        if instruction.startswith("press"):
            return "keyboard_press"

        if instruction.startswith("hotkey"):
            return "hotkey"

        if "screenshot" in instruction:
            return "screenshot"

        if "clipboard" in instruction:
            if any(
                word in instruction
                for word in (
                    "set",
                    "copy",
                    "write",
                )
            ):
                return "clipboard_set"

            return "clipboard_get"

        if instruction.startswith("open application"):
            return "open_application"

        if instruction.startswith("launch"):
            return "open_application"

        if instruction.startswith("close application"):
            return "close_application"

        if "minimize window" in instruction:
            return "window_minimize"

        if "maximize window" in instruction:
            return "window_maximize"

        if "close window" in instruction:
            return "window_close"

        if "focus window" in instruction:
            return "window_focus"

        if "volume" in instruction:
            return "volume"

        if "brightness" in instruction:
            return "brightness"

        if "screen info" in instruction:
            return "screen_info"

        if "show desktop" in instruction:
            return "desktop"

        if "process list" in instruction:
            return "process_list"

        if "kill process" in instruction:
            return "process_kill"

        if "system command" in instruction:
            return "system_command"

        if instruction.startswith("open url"):
            return "launch_url"

        if instruction.startswith("open file"):
            return "open_file"

        return "unknown"

    # ========================================================================
    # MOUSE
    # ========================================================================

    def mouse_move(
        self,
        x: int,
        y: int,
        duration: float = 0.0,
        **_: Any,
    ) -> Dict[str, Any]:

        if self.mouse_controller is not None:

            for method_name in (
                "move",
                "move_to",
                "move_mouse",
            ):

                method = getattr(
                    self.mouse_controller,
                    method_name,
                    None,
                )

                if callable(method):

                    result = method(
                        x,
                        y,
                        duration=duration,
                    )

                    return {
                        "success": True,
                        "action": "mouse_move",
                        "result": result,
                    }

        try:
            import pyautogui

            pyautogui.moveTo(
                x,
                y,
                duration=duration,
            )

            return {
                "success": True,
                "action": "mouse_move",
                "x": x,
                "y": y,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    def mouse_click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.1,
        **_: Any,
    ) -> Dict[str, Any]:

        try:
            import pyautogui

            if x is not None and y is not None:
                pyautogui.moveTo(x, y)

            pyautogui.click(
                button=button,
                clicks=clicks,
                interval=interval,
            )

            return {
                "success": True,
                "action": "mouse_click",
                "x": x,
                "y": y,
                "button": button,
                "clicks": clicks,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    def mouse_double_click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        **kwargs: Any,
    ) -> Dict[str, Any]:

        return self.mouse_click(
            x=x,
            y=y,
            button=button,
            clicks=2,
            **kwargs,
        )

    def mouse_scroll(
        self,
        amount: int,
        x: Optional[int] = None,
        y: Optional[int] = None,
        **_: Any,
    ) -> Dict[str, Any]:

        try:
            import pyautogui

            if x is not None and y is not None:
                pyautogui.moveTo(x, y)

            pyautogui.scroll(amount)

            return {
                "success": True,
                "action": "mouse_scroll",
                "amount": amount,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # KEYBOARD
    # ========================================================================

    def keyboard_type(
        self,
        text: str,
        interval: float = 0.01,
        **_: Any,
    ) -> Dict[str, Any]:

        try:
            import pyautogui

            pyautogui.write(
                text,
                interval=interval,
            )

            return {
                "success": True,
                "action": "keyboard_type",
                "length": len(text),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    def keyboard_press(
        self,
        key: str,
        presses: int = 1,
        interval: float = 0.05,
        **_: Any,
    ) -> Dict[str, Any]:

        try:
            import pyautogui

            pyautogui.press(
                key,
                presses=presses,
                interval=interval,
            )

            return {
                "success": True,
                "action": "keyboard_press",
                "key": key,
                "presses": presses,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    def hotkey(
        self,
        *keys: str,
        **_: Any,
    ) -> Dict[str, Any]:

        try:
            import pyautogui

            pyautogui.hotkey(
                *keys
            )

            return {
                "success": True,
                "action": "hotkey",
                "keys": list(keys),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # SCREENSHOTS
    # ========================================================================

    def screenshot(
        self,
        path: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:

        try:

            if self.screenshot_controller is not None:

                for method_name in (
                    "capture",
                    "screenshot",
                    "take_screenshot",
                ):

                    method = getattr(
                        self.screenshot_controller,
                        method_name,
                        None,
                    )

                    if callable(method):

                        result = (
                            method(path)
                            if path
                            else method()
                        )

                        return {
                            "success": True,
                            "action": "screenshot",
                            "path": str(result)
                            if result
                            else path,
                        }

            import pyautogui

            image = pyautogui.screenshot()

            if path is None:
                path = str(
                    Path.cwd()
                    / "renix_screenshot.png"
                )

            image.save(path)

            return {
                "success": True,
                "action": "screenshot",
                "path": str(path),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # CLIPBOARD
    # ========================================================================

    def clipboard_get(
        self,
        **_: Any,
    ) -> Dict[str, Any]:

        try:

            if self.clipboard_controller is not None:

                for method_name in (
                    "get",
                    "read",
                    "get_text",
                ):

                    method = getattr(
                        self.clipboard_controller,
                        method_name,
                        None,
                    )

                    if callable(method):

                        value = method()

                        return {
                            "success": True,
                            "action": "clipboard_get",
                            "text": value,
                        }

            import tkinter as tk

            root = tk.Tk()
            root.withdraw()

            try:
                value = root.clipboard_get()
            finally:
                root.destroy()

            return {
                "success": True,
                "action": "clipboard_get",
                "text": value,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    def clipboard_set(
        self,
        text: str,
        **_: Any,
    ) -> Dict[str, Any]:

        try:

            if self.clipboard_controller is not None:

                for method_name in (
                    "set",
                    "write",
                    "set_text",
                ):

                    method = getattr(
                        self.clipboard_controller,
                        method_name,
                        None,
                    )

                    if callable(method):

                        method(text)

                        return {
                            "success": True,
                            "action": "clipboard_set",
                        }

            import tkinter as tk

            root = tk.Tk()
            root.withdraw()

            try:
                root.clipboard_clear()
                root.clipboard_append(text)
                root.update()
            finally:
                root.destroy()

            return {
                "success": True,
                "action": "clipboard_set",
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # APPLICATIONS
    # ========================================================================

    def open_application(
        self,
        application: str,
        **_: Any,
    ) -> Dict[str, Any]:

        try:

            if self.application_controller is not None:

                for method_name in (
                    "open",
                    "launch",
                    "start",
                    "open_application",
                ):

                    method = getattr(
                        self.application_controller,
                        method_name,
                        None,
                    )

                    if callable(method):

                        result = method(
                            application
                        )

                        return {
                            "success": True,
                            "action": "open_application",
                            "application": application,
                            "result": result,
                        }

            if self.system == "windows":

                subprocess.Popen(
                    [
                        "cmd",
                        "/c",
                        "start",
                        "",
                        application,
                    ],
                    shell=False,
                )

            elif self.system == "darwin":

                subprocess.Popen(
                    [
                        "open",
                        application,
                    ]
                )

            else:

                subprocess.Popen(
                    [
                        application
                    ]
                )

            return {
                "success": True,
                "action": "open_application",
                "application": application,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
                "application": application,
            }

    def close_application(
        self,
        application: str,
        **_: Any,
    ) -> Dict[str, Any]:

        try:

            if self.application_controller is not None:

                for method_name in (
                    "close",
                    "terminate",
                    "stop",
                    "close_application",
                ):

                    method = getattr(
                        self.application_controller,
                        method_name,
                        None,
                    )

                    if callable(method):

                        result = method(
                            application
                        )

                        return {
                            "success": True,
                            "action": "close_application",
                            "application": application,
                            "result": result,
                        }

            if self.system == "windows":

                subprocess.run(
                    [
                        "taskkill",
                        "/IM",
                        application,
                        "/F",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )

            else:

                subprocess.run(
                    [
                        "pkill",
                        "-f",
                        application,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )

            return {
                "success": True,
                "action": "close_application",
                "application": application,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # WINDOWS
    # ========================================================================

    def window_minimize(
        self,
        title: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:

        return self._window_action(
            "minimize",
            title,
        )

    def window_maximize(
        self,
        title: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:

        return self._window_action(
            "maximize",
            title,
        )

    def window_close(
        self,
        title: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:

        return self._window_action(
            "close",
            title,
        )

    def window_focus(
        self,
        title: str,
        **_: Any,
    ) -> Dict[str, Any]:

        return self._window_action(
            "focus",
            title,
        )

    def _window_action(
        self,
        action: str,
        title: Optional[str],
    ) -> Dict[str, Any]:

        try:

            if self.window_controller is not None:

                for method_name in (
                    action,
                    f"{action}_window",
                ):

                    method = getattr(
                        self.window_controller,
                        method_name,
                        None,
                    )

                    if callable(method):

                        result = (
                            method(title)
                            if title
                            else method()
                        )

                        return {
                            "success": True,
                            "action": action,
                            "title": title,
                            "result": result,
                        }

            if self.system == "windows":

                return self._windows_native_action(
                    action,
                    title,
                )

            return {
                "success": False,
                "error": (
                    "Native window control is not "
                    "implemented for this platform."
                ),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    def _windows_native_action(
        self,
        action: str,
        title: Optional[str],
    ) -> Dict[str, Any]:

        try:

            import pyautogui

            if action == "minimize":
                pyautogui.hotkey(
                    "alt",
                    "space",
                )
                pyautogui.press(
                    "n"
                )

            elif action == "maximize":
                pyautogui.hotkey(
                    "alt",
                    "space",
                )
                pyautogui.press(
                    "x"
                )

            elif action == "close":
                pyautogui.hotkey(
                    "alt",
                    "f4",
                )

            elif action == "focus":

                if not title:
                    return {
                        "success": False,
                        "error": (
                            "Window title is required."
                        ),
                    }

                try:

                    import pygetwindow as gw

                    windows = gw.getWindowsWithTitle(
                        title
                    )

                    if not windows:

                        return {
                            "success": False,
                            "error": (
                                f"Window not found: {title}"
                            ),
                        }

                    windows[0].activate()

                except Exception as exc:

                    return {
                        "success": False,
                        "error": str(exc),
                    }

            else:

                return {
                    "success": False,
                    "error": (
                        f"Unsupported window action: {action}"
                    ),
                }

            return {
                "success": True,
                "action": action,
                "title": title,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # SYSTEM CONTROLS
    # ========================================================================

    def set_volume(
        self,
        level: Optional[int] = None,
        change: Optional[int] = None,
        mute: Optional[bool] = None,
        **_: Any,
    ) -> Dict[str, Any]:

        try:

            if self.volume_controller is not None:

                if mute is True:

                    for method_name in (
                        "mute",
                        "toggle_mute",
                    ):

                        method = getattr(
                            self.volume_controller,
                            method_name,
                            None,
                        )

                        if callable(method):

                            method()

                            return {
                                "success": True,
                                "action": "volume",
                                "mute": True,
                            }

                if level is not None:

                    for method_name in (
                        "set",
                        "set_volume",
                    ):

                        method = getattr(
                            self.volume_controller,
                            method_name,
                            None,
                        )

                        if callable(method):

                            method(level)

                            return {
                                "success": True,
                                "action": "volume",
                                "level": level,
                            }

                if change is not None:

                    for method_name in (
                        "change",
                        "adjust",
                    ):

                        method = getattr(
                            self.volume_controller,
                            method_name,
                            None,
                        )

                        if callable(method):

                            method(change)

                            return {
                                "success": True,
                                "action": "volume",
                                "change": change,
                            }

            return {
                "success": False,
                "error": (
                    "Volume controller is unavailable."
                ),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    def set_brightness(
        self,
        level: int,
        **_: Any,
    ) -> Dict[str, Any]:

        try:

            if self.brightness_controller is None:

                return {
                    "success": False,
                    "error": (
                        "Brightness controller is unavailable."
                    ),
                }

            for method_name in (
                "set",
                "set_brightness",
            ):

                method = getattr(
                    self.brightness_controller,
                    method_name,
                    None,
                )

                if callable(method):

                    result = method(level)

                    return {
                        "success": True,
                        "action": "brightness",
                        "level": level,
                        "result": result,
                    }

            return {
                "success": False,
                "error": (
                    "Brightness method unavailable."
                ),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # SCREEN / DISPLAY
    # ========================================================================

    def get_screen_info(
        self,
        **_: Any,
    ) -> Dict[str, Any]:

        try:

            if self.screen_controller is not None:

                for method_name in (
                    "get_info",
                    "info",
                    "screen_info",
                ):

                    method = getattr(
                        self.screen_controller,
                        method_name,
                        None,
                    )

                    if callable(method):

                        return {
                            "success": True,
                            "action": "screen_info",
                            "data": method(),
                        }

            import pyautogui

            width, height = (
                pyautogui.size()
            )

            return {
                "success": True,
                "action": "screen_info",
                "width": width,
                "height": height,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    def show_desktop(
        self,
        **_: Any,
    ) -> Dict[str, Any]:

        try:

            if self.desktop_controller is not None:

                for method_name in (
                    "show",
                    "show_desktop",
                    "minimize_all",
                ):

                    method = getattr(
                        self.desktop_controller,
                        method_name,
                        None,
                    )

                    if callable(method):

                        method()

                        return {
                            "success": True,
                            "action": "desktop",
                        }

            if self.system == "windows":

                import pyautogui

                pyautogui.hotkey(
                    "win",
                    "d",
                )

                return {
                    "success": True,
                    "action": "desktop",
                }

            return {
                "success": False,
                "error": (
                    "Desktop control unavailable."
                ),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # PROCESSES
    # ========================================================================

    def list_processes(
        self,
        **_: Any,
    ) -> Dict[str, Any]:

        try:

            if self.process_controller is not None:

                for method_name in (
                    "list",
                    "list_processes",
                    "get_processes",
                ):

                    method = getattr(
                        self.process_controller,
                        method_name,
                        None,
                    )

                    if callable(method):

                        return {
                            "success": True,
                            "action": "process_list",
                            "processes": method(),
                        }

            try:

                import psutil

                processes = []

                for process in psutil.process_iter(
                    [
                        "pid",
                        "name",
                        "status",
                    ]
                ):

                    try:

                        processes.append(
                            process.info
                        )

                    except (
                        psutil.NoSuchProcess,
                        psutil.AccessDenied,
                    ):

                        continue

                return {
                    "success": True,
                    "action": "process_list",
                    "processes": processes,
                }

            except ImportError:

                return {
                    "success": False,
                    "error": (
                        "psutil is required for process listing."
                    ),
                }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    def kill_process(
        self,
        pid: Optional[int] = None,
        name: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:

        try:

            if pid is None and not name:

                return {
                    "success": False,
                    "error": (
                        "Either pid or name is required."
                    ),
                }

            if self.process_controller is not None:

                for method_name in (
                    "kill",
                    "terminate",
                    "kill_process",
                ):

                    method = getattr(
                        self.process_controller,
                        method_name,
                        None,
                    )

                    if callable(method):

                        argument = (
                            pid
                            if pid is not None
                            else name
                        )

                        result = method(
                            argument
                        )

                        return {
                            "success": True,
                            "action": "process_kill",
                            "result": result,
                        }

            import psutil

            if pid is not None:

                process = psutil.Process(
                    int(pid)
                )

                process.terminate()

            else:

                for process in psutil.process_iter(
                    ["pid", "name"]
                ):

                    try:

                        process_name = (
                            process.info.get(
                                "name"
                            )
                        )

                        if (
                            process_name
                            and process_name.lower()
                            == name.lower()
                        ):

                            process.terminate()

                    except (
                        psutil.NoSuchProcess,
                        psutil.AccessDenied,
                    ):

                        continue

            return {
                "success": True,
                "action": "process_kill",
                "pid": pid,
                "name": name,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # SYSTEM COMMAND
    # ========================================================================

    def execute_system_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 120,
        **_: Any,
    ) -> Dict[str, Any]:

        if not command.strip():

            return {
                "success": False,
                "error": "Command cannot be empty.",
            }

        try:

            completed = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            return {
                "success": (
                    completed.returncode == 0
                ),
                "action": "system_command",
                "command": command,
                "return_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }

        except subprocess.TimeoutExpired:

            return {
                "success": False,
                "error": (
                    "Command timed out."
                ),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # URL / FILE OPENING
    # ========================================================================

    def launch_url(
        self,
        url: str,
        **_: Any,
    ) -> Dict[str, Any]:

        if not url.strip():

            return {
                "success": False,
                "error": "URL cannot be empty.",
            }

        try:

            if self.system == "windows":

                os.startfile(url)

            elif self.system == "darwin":

                subprocess.Popen(
                    [
                        "open",
                        url,
                    ]
                )

            else:

                subprocess.Popen(
                    [
                        "xdg-open",
                        url,
                    ]
                )

            return {
                "success": True,
                "action": "launch_url",
                "url": url,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    def open_file(
        self,
        path: str,
        **_: Any,
    ) -> Dict[str, Any]:

        file_path = Path(
            path
        ).expanduser()

        if not file_path.exists():

            return {
                "success": False,
                "error": (
                    f"File does not exist: {file_path}"
                ),
            }

        try:

            if self.system == "windows":

                os.startfile(
                    str(file_path)
                )

            elif self.system == "darwin":

                subprocess.Popen(
                    [
                        "open",
                        str(file_path),
                    ]
                )

            else:

                subprocess.Popen(
                    [
                        "xdg-open",
                        str(file_path),
                    ]
                )

            return {
                "success": True,
                "action": "open_file",
                "path": str(file_path),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "ComputerAgent",
]


