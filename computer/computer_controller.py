"""
RENIX Computer Controller
=========================

Central controller for computer-level operations.

This module acts as the bridge between RENIX's core/AI systems
and the lower-level computer modules:

    mouse.py
    keyboard.py
    screen.py
    windows.py
    applications.py
    clipboard.py
    screenshots.py
    screen_reader.py
    desktop.py
    system_settings.py
    volume.py
    brightness.py
    display.py
    process_manager.py

The controller intentionally keeps orchestration logic here and
delegates actual operations to specialized modules.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


logger = logging.getLogger(__name__)


@dataclass
class ComputerAction:
    """
    Represents an action requested by RENIX.
    """

    action_id: str
    action_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(
        default_factory=time.time
    )

    status: str = "pending"
    result: Any = None
    error: Optional[str] = None


class ComputerController:
    """
    High-level computer controller.

    Example:

        controller = ComputerController()

        controller.move_mouse(500, 300)
        controller.click()

        controller.type_text("Hello RENIX")
        controller.press_key("enter")
    """

    def __init__(
        self,
        mouse: Any = None,
        keyboard: Any = None,
        screen: Any = None,
        windows: Any = None,
        applications: Any = None,
        clipboard: Any = None,
        screenshots: Any = None,
        screen_reader: Any = None,
        desktop: Any = None,
        system_settings: Any = None,
        volume: Any = None,
        brightness: Any = None,
        display: Any = None,
        process_manager: Any = None,
    ) -> None:

        self.mouse = mouse
        self.keyboard = keyboard
        self.screen = screen
        self.windows = windows
        self.applications = applications
        self.clipboard = clipboard
        self.screenshots = screenshots
        self.screen_reader = screen_reader
        self.desktop = desktop
        self.system_settings = system_settings
        self.volume = volume
        self.brightness = brightness
        self.display = display
        self.process_manager = process_manager

        self.enabled = True

        self.lock = threading.RLock()

        self.action_counter = 0

        self.action_history: list[ComputerAction] = []

        self.max_history = 500

        self.event_handlers: dict[
            str,
            list[Callable[..., Any]],
        ] = {}

        logger.info(
            "RENIX ComputerController initialized"
        )

    # ==========================================================
    # GENERAL CONTROL
    # ==========================================================

    def enable(self) -> None:
        """
        Enable computer control.
        """

        with self.lock:
            self.enabled = True

        logger.info(
            "Computer control enabled"
        )

    def disable(self) -> None:
        """
        Disable computer control.
        """

        with self.lock:
            self.enabled = False

        logger.info(
            "Computer control disabled"
        )

    def is_enabled(self) -> bool:
        """
        Return whether computer control is enabled.
        """

        return self.enabled

    def _check_enabled(self) -> None:

        if not self.enabled:
            raise RuntimeError(
                "RENIX computer control is disabled."
            )

    # ==========================================================
    # ACTION MANAGEMENT
    # ==========================================================

    def _create_action(
        self,
        action_type: str,
        parameters: Optional[
            dict[str, Any]
        ] = None,
    ) -> ComputerAction:

        self.action_counter += 1

        action = ComputerAction(
            action_id=(
                f"computer-{self.action_counter}"
            ),
            action_type=action_type,
            parameters=parameters or {},
        )

        self.action_history.append(action)

        if (
            len(self.action_history)
            > self.max_history
        ):
            self.action_history.pop(0)

        return action

    def _execute(
        self,
        action_type: str,
        callback: Callable[[], Any],
        parameters: Optional[
            dict[str, Any]
        ] = None,
    ) -> Any:

        self._check_enabled()

        action = self._create_action(
            action_type,
            parameters,
        )

        self._emit(
            "action_started",
            action,
        )

        try:

            result = callback()

            action.status = "completed"
            action.result = result

            self._emit(
                "action_completed",
                action,
            )

            return result

        except Exception as exc:

            action.status = "failed"
            action.error = str(exc)

            logger.exception(
                "Computer action failed: %s",
                action_type,
            )

            self._emit(
                "action_failed",
                action,
            )

            raise

    # ==========================================================
    # MOUSE
    # ==========================================================

    def move_mouse(
        self,
        x: int,
        y: int,
        duration: float = 0.0,
    ) -> Any:

        if self.mouse is None:
            raise RuntimeError(
                "Mouse controller is not configured."
            )

        method = getattr(
            self.mouse,
            "move",
            None,
        )

        if method is None:
            raise AttributeError(
                "Mouse controller does not provide move()."
            )

        return self._execute(
            "mouse_move",
            lambda: method(
                x,
                y,
                duration=duration,
            ),
            {
                "x": x,
                "y": y,
                "duration": duration,
            },
        )

    def click(
        self,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.0,
    ) -> Any:

        if self.mouse is None:
            raise RuntimeError(
                "Mouse controller is not configured."
            )

        method = getattr(
            self.mouse,
            "click",
            None,
        )

        if method is None:
            raise AttributeError(
                "Mouse controller does not provide click()."
            )

        return self._execute(
            "mouse_click",
            lambda: method(
                button=button,
                clicks=clicks,
                interval=interval,
            ),
            {
                "button": button,
                "clicks": clicks,
                "interval": interval,
            },
        )

    def mouse_down(
        self,
        button: str = "left",
    ) -> Any:

        if self.mouse is None:
            raise RuntimeError(
                "Mouse controller is not configured."
            )

        method = getattr(
            self.mouse,
            "button_down",
            None,
        )

        if method is None:
            raise AttributeError(
                "Mouse controller does not provide button_down()."
            )

        return self._execute(
            "mouse_down",
            lambda: method(button),
            {"button": button},
        )

    def mouse_up(
        self,
        button: str = "left",
    ) -> Any:

        if self.mouse is None:
            raise RuntimeError(
                "Mouse controller is not configured."
            )

        method = getattr(
            self.mouse,
            "button_up",
            None,
        )

        if method is None:
            raise AttributeError(
                "Mouse controller does not provide button_up()."
            )

        return self._execute(
            "mouse_up",
            lambda: method(button),
            {"button": button},
        )

    def scroll(
        self,
        amount: int,
    ) -> Any:

        if self.mouse is None:
            raise RuntimeError(
                "Mouse controller is not configured."
            )

        method = getattr(
            self.mouse,
            "scroll",
            None,
        )

        if method is None:
            raise AttributeError(
                "Mouse controller does not provide scroll()."
            )

        return self._execute(
            "mouse_scroll",
            lambda: method(amount),
            {"amount": amount},
        )

    # ==========================================================
    # KEYBOARD
    # ==========================================================

    def press_key(
        self,
        key: str,
    ) -> Any:

        if self.keyboard is None:
            raise RuntimeError(
                "Keyboard controller is not configured."
            )

        method = getattr(
            self.keyboard,
            "press",
            None,
        )

        if method is None:
            raise AttributeError(
                "Keyboard controller does not provide press()."
            )

        return self._execute(
            "key_press",
            lambda: method(key),
            {"key": key},
        )

    def release_key(
        self,
        key: str,
    ) -> Any:

        if self.keyboard is None:
            raise RuntimeError(
                "Keyboard controller is not configured."
            )

        method = getattr(
            self.keyboard,
            "release",
            None,
        )

        if method is None:
            raise AttributeError(
                "Keyboard controller does not provide release()."
            )

        return self._execute(
            "key_release",
            lambda: method(key),
            {"key": key},
        )

    def type_text(
        self,
        text: str,
        interval: float = 0.0,
    ) -> Any:

        if self.keyboard is None:
            raise RuntimeError(
                "Keyboard controller is not configured."
            )

        method = getattr(
            self.keyboard,
            "type_text",
            None,
        )

        if method is None:
            raise AttributeError(
                "Keyboard controller does not provide type_text()."
            )

        return self._execute(
            "type_text",
            lambda: method(
                text,
                interval=interval,
            ),
            {
                "text": text,
                "interval": interval,
            },
        )

    def hotkey(
        self,
        *keys: str,
    ) -> Any:

        if self.keyboard is None:
            raise RuntimeError(
                "Keyboard controller is not configured."
            )

        method = getattr(
            self.keyboard,
            "hotkey",
            None,
        )

        if method is None:
            raise AttributeError(
                "Keyboard controller does not provide hotkey()."
            )

        return self._execute(
            "keyboard_hotkey",
            lambda: method(*keys),
            {"keys": list(keys)},
        )

    # ==========================================================
    # SCREEN
    # ==========================================================

    def get_screen_size(self) -> Any:

        if self.screen is None:
            raise RuntimeError(
                "Screen controller is not configured."
            )

        method = getattr(
            self.screen,
            "get_size",
            None,
        )

        if method is None:
            raise AttributeError(
                "Screen controller does not provide get_size()."
            )

        return method()

    def get_mouse_position(self) -> Any:

        if self.mouse is None:
            raise RuntimeError(
                "Mouse controller is not configured."
            )

        method = getattr(
            self.mouse,
            "get_position",
            None,
        )

        if method is None:
            raise AttributeError(
                "Mouse controller does not provide get_position()."
            )

        return method()

    # ==========================================================
    # CLIPBOARD
    # ==========================================================

    def copy(self) -> Any:

        return self.hotkey(
            "ctrl",
            "c",
        )

    def paste(self) -> Any:

        return self.hotkey(
            "ctrl",
            "v",
        )

    def cut(self) -> Any:

        return self.hotkey(
            "ctrl",
            "x",
        )

    def select_all(self) -> Any:

        return self.hotkey(
            "ctrl",
            "a",
        )

    def get_clipboard(self) -> Any:

        if self.clipboard is None:
            raise RuntimeError(
                "Clipboard manager is not configured."
            )

        method = getattr(
            self.clipboard,
            "get",
            None,
        )

        if method is None:
            raise AttributeError(
                "Clipboard manager does not provide get()."
            )

        return method()

    def set_clipboard(
        self,
        text: str,
    ) -> Any:

        if self.clipboard is None:
            raise RuntimeError(
                "Clipboard manager is not configured."
            )

        method = getattr(
            self.clipboard,
            "set",
            None,
        )

        if method is None:
            raise AttributeError(
                "Clipboard manager does not provide set()."
            )

        return method(text)

    # ==========================================================
    # APPLICATIONS
    # ==========================================================

    def open_application(
        self,
        name: str,
        *args: str,
    ) -> Any:

        if self.applications is None:
            raise RuntimeError(
                "Application manager is not configured."
            )

        method = getattr(
            self.applications,
            "launch",
            None,
        )

        if method is None:
            raise AttributeError(
                "Application manager does not provide launch()."
            )

        return self._execute(
            "application_launch",
            lambda: method(
                name,
                *args,
            ),
            {
                "name": name,
                "args": list(args),
            },
        )

    def close_application(
        self,
        name: str,
    ) -> Any:

        if self.applications is None:
            raise RuntimeError(
                "Application manager is not configured."
            )

        method = getattr(
            self.applications,
            "close",
            None,
        )

        if method is None:
            raise AttributeError(
                "Application manager does not provide close()."
            )

        return self._execute(
            "application_close",
            lambda: method(name),
            {"name": name},
        )

    # ==========================================================
    # WINDOWS
    # ==========================================================

    def get_windows(self) -> Any:

        if self.windows is None:
            raise RuntimeError(
                "Window manager is not configured."
            )

        method = getattr(
            self.windows,
            "list_windows",
            None,
        )

        if method is None:
            raise AttributeError(
                "Window manager does not provide list_windows()."
            )

        return method()

    def focus_window(
        self,
        title: str,
    ) -> Any:

        if self.windows is None:
            raise RuntimeError(
                "Window manager is not configured."
            )

        method = getattr(
            self.windows,
            "focus",
            None,
        )

        if method is None:
            raise AttributeError(
                "Window manager does not provide focus()."
            )

        return self._execute(
            "window_focus",
            lambda: method(title),
            {"title": title},
        )

    # ==========================================================
    # SCREENSHOTS
    # ==========================================================

    def screenshot(
        self,
        path: Optional[str] = None,
    ) -> Any:

        if self.screenshots is None:
            raise RuntimeError(
                "Screenshot manager is not configured."
            )

        method = getattr(
            self.screenshots,
            "capture",
            None,
        )

        if method is None:
            raise AttributeError(
                "Screenshot manager does not provide capture()."
            )

        return self._execute(
            "screenshot",
            lambda: method(path),
            {"path": path},
        )

    # ==========================================================
    # SYSTEM
    # ==========================================================

    def set_volume(
        self,
        value: float,
    ) -> Any:

        if self.volume is None:
            raise RuntimeError(
                "Volume controller is not configured."
            )

        method = getattr(
            self.volume,
            "set_volume",
            None,
        )

        if method is None:
            raise AttributeError(
                "Volume controller does not provide set_volume()."
            )

        return self._execute(
            "set_volume",
            lambda: method(value),
            {"value": value},
        )

    def set_brightness(
        self,
        value: float,
    ) -> Any:

        if self.brightness is None:
            raise RuntimeError(
                "Brightness controller is not configured."
            )

        method = getattr(
            self.brightness,
            "set_brightness",
            None,
        )

        if method is None:
            raise AttributeError(
                "Brightness controller does not provide set_brightness()."
            )

        return self._execute(
            "set_brightness",
            lambda: method(value),
            {"value": value},
        )

    # ==========================================================
    # PROCESS MANAGEMENT
    # ==========================================================

    def list_processes(self) -> Any:

        if self.process_manager is None:
            raise RuntimeError(
                "Process manager is not configured."
            )

        method = getattr(
            self.process_manager,
            "list_processes",
            None,
        )

        if method is None:
            raise AttributeError(
                "Process manager does not provide list_processes()."
            )

        return method()

    def terminate_process(
        self,
        process_id: int,
    ) -> Any:

        if self.process_manager is None:
            raise RuntimeError(
                "Process manager is not configured."
            )

        method = getattr(
            self.process_manager,
            "terminate",
            None,
        )

        if method is None:
            raise AttributeError(
                "Process manager does not provide terminate()."
            )

        return self._execute(
            "process_terminate",
            lambda: method(process_id),
            {
                "process_id": process_id
            },
        )

    # ==========================================================
    # EVENT SYSTEM
    # ==========================================================

    def on(
        self,
        event: str,
        handler: Callable[..., Any],
    ) -> None:

        self.event_handlers.setdefault(
            event,
            [],
        ).append(handler)

    def off(
        self,
        event: str,
        handler: Callable[..., Any],
    ) -> None:

        handlers = self.event_handlers.get(
            event,
            [],
        )

        if handler in handlers:
            handlers.remove(handler)

    def _emit(
        self,
        event: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        handlers = self.event_handlers.get(
            event,
            [],
        )

        for handler in list(handlers):

            try:
                handler(
                    *args,
                    **kwargs,
                )

            except Exception:

                logger.exception(
                    "Computer event handler failed: %s",
                    event,
                )

    # ==========================================================
    # HISTORY
    # ==========================================================

    def get_action_history(
        self,
        limit: Optional[int] = None,
    ) -> list[ComputerAction]:

        history = list(
            self.action_history
        )

        if limit is not None:
            return history[-limit:]

        return history

    def clear_action_history(self) -> None:

        with self.lock:
            self.action_history.clear()

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "mouse": self.mouse is not None,
            "keyboard": self.keyboard is not None,
            "screen": self.screen is not None,
            "windows": self.windows is not None,
            "applications": (
                self.applications is not None
            ),
            "clipboard": (
                self.clipboard is not None
            ),
            "screenshots": (
                self.screenshots is not None
            ),
            "screen_reader": (
                self.screen_reader is not None
            ),
            "desktop": (
                self.desktop is not None
            ),
            "system_settings": (
                self.system_settings is not None
            ),
            "volume": self.volume is not None,
            "brightness": (
                self.brightness is not None
            ),
            "display": self.display is not None,
            "process_manager": (
                self.process_manager is not None
            ),
            "actions": len(
                self.action_history
            ),
        }

    def reset(self) -> None:

        with self.lock:

            self.action_history.clear()

            self.action_counter = 0

        logger.info(
            "Computer controller reset"
        )


