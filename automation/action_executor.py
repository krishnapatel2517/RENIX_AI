"""
RENIX Action Executor
=====================

Executes individual RENIX automation actions.

The ActionExecutor acts as the bridge between workflows and
RENIX services such as:

    - Computer control
    - Files
    - Browser
    - Voice
    - Notifications
    - Smart home
    - Media
    - Coding
    - Education
    - Personal tasks
    - System operations

Services are injected into the executor so the automation layer
does not depend on concrete implementations.
"""

from __future__ import annotations

import logging
import subprocess
import time
import webbrowser

from typing import Any, Callable

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Central executor for RENIX automation actions."""

    def __init__(
        self,
        *,
        services: dict[str, Any] | None = None,
        event_bus: Any = None,
        security_manager: Any = None,
        context: dict[str, Any] | None = None,
    ) -> None:

        self.services: dict[str, Any] = dict(
            services or {}
        )

        self.event_bus = event_bus
        self.security_manager = security_manager

        self.context: dict[str, Any] = dict(
            context or {}
        )

        self._custom_actions: dict[
            str,
            Callable[..., Any],
        ] = {}

    # ========================================================
    # SERVICE MANAGEMENT
    # ========================================================

    def register_service(
        self,
        name: str,
        service: Any,
    ) -> None:
        """Register a RENIX service."""

        name = name.strip().lower()

        if not name:
            raise ValueError(
                "Service name cannot be empty."
            )

        if service is None:
            raise ValueError(
                "Service cannot be None."
            )

        self.services[name] = service

    def unregister_service(
        self,
        name: str,
    ) -> bool:
        """Remove a registered service."""

        name = name.strip().lower()

        if name not in self.services:
            return False

        del self.services[name]

        return True

    def get_service(
        self,
        name: str,
    ) -> Any:
        """Return a registered service."""

        name = name.strip().lower()

        service = self.services.get(
            name
        )

        if service is None:
            raise KeyError(
                f"RENIX service not registered: {name}"
            )

        return service

    # ========================================================
    # CUSTOM ACTIONS
    # ========================================================

    def register_action(
        self,
        action_type: str,
        handler: Callable[..., Any],
    ) -> None:
        """Register a custom action handler."""

        action_type = action_type.strip().lower()

        if not action_type:
            raise ValueError(
                "Action type cannot be empty."
            )

        if not callable(handler):
            raise TypeError(
                "Action handler must be callable."
            )

        self._custom_actions[
            action_type
        ] = handler

    def unregister_action(
        self,
        action_type: str,
    ) -> bool:
        """Remove a custom action handler."""

        action_type = action_type.strip().lower()

        if action_type not in self._custom_actions:
            return False

        del self._custom_actions[
            action_type
        ]

        return True

    # ========================================================
    # MAIN EXECUTION
    # ========================================================

    def execute(
        self,
        action: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """
        Execute an automation action.

        Example:

        {
            "type": "notification",
            "message": "Good morning!"
        }
        """

        if not isinstance(
            action,
            dict,
        ):
            raise TypeError(
                "action must be a dictionary."
            )

        action_type = str(
            action.get(
                "type",
                "",
            )
        ).strip().lower()

        if not action_type:
            raise ValueError(
                "Action type is required."
            )

        if action.get(
            "enabled",
            True,
        ) is False:
            return {
                "status": "skipped",
                "reason": "disabled",
                "action": action_type,
            }

        merged_context = dict(
            self.context
        )

        merged_context.update(
            context or {}
        )

        self._check_security(
            action,
            merged_context,
        )

        self._emit(
            "automation.action.started",
            {
                "type": action_type,
                "action": action,
            },
        )

        try:

            handler = self._custom_actions.get(
                action_type
            )

            if handler is not None:

                result = handler(
                    action,
                    merged_context,
                )

            else:

                result = self._dispatch(
                    action_type,
                    action,
                    merged_context,
                )

            self._emit(
                "automation.action.completed",
                {
                    "type": action_type,
                    "result": result,
                },
            )

            return result

        except Exception as exc:

            logger.exception(
                "Automation action failed: %s",
                action_type,
            )

            self._emit(
                "automation.action.failed",
                {
                    "type": action_type,
                    "error": str(exc),
                },
            )

            raise

    # ========================================================
    # DISPATCH
    # ========================================================

    def _dispatch(
        self,
        action_type: str,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:

        handlers = {
            "delay": self._delay,
            "wait": self._delay,
            "sleep": self._delay,

            "notification": self._notification,
            "notify": self._notification,

            "speak": self._speak,
            "say": self._speak,

            "open_url": self._open_url,
            "browser": self._open_url,

            "run_command": self._run_command,
            "command": self._run_command,

            "set_variable": self._set_variable,

            "set_context": self._set_variable,

            "computer": self._computer,
            "computer_action": self._computer,

            "file": self._file,
            "file_action": self._file,

            "browser_action": self._browser,

            "smart_home": self._smart_home,

            "media": self._media,

            "task": self._task,
            "create_task": self._task,

            "calendar": self._calendar,

            "email": self._email,

            "coding": self._coding,

            "system": self._system,
        }

        handler = handlers.get(
            action_type
        )

        if handler is None:
            raise ValueError(
                f"Unknown automation action: {action_type}"
            )

        return handler(
            action,
            context,
        )

    # ========================================================
    # BASIC ACTIONS
    # ========================================================

    def _delay(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        seconds = float(
            action.get(
                "seconds",
                action.get(
                    "duration",
                    0,
                ),
            )
        )

        if seconds < 0:
            raise ValueError(
                "Delay cannot be negative."
            )

        time.sleep(
            seconds
        )

        return {
            "status": "completed",
            "action": "delay",
            "seconds": seconds,
        }

    def _set_variable(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        key = str(
            action.get(
                "key",
                "",
            )
        ).strip()

        if not key:
            raise ValueError(
                "Variable key is required."
            )

        value = action.get(
            "value"
        )

        context[key] = value

        return {
            "status": "completed",
            "key": key,
            "value": value,
        }

    # ========================================================
    # NOTIFICATIONS
    # ========================================================

    def _notification(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:

        message = self._resolve(
            action.get(
                "message",
                "",
            ),
            context,
        )

        title = self._resolve(
            action.get(
                "title",
                "RENIX",
            ),
            context,
        )

        service = self.services.get(
            "notifications"
        )

        if service is None:
            logger.info(
                "[NOTIFICATION] %s: %s",
                title,
                message,
            )

            return {
                "status": "logged",
                "title": title,
                "message": message,
            }

        method = getattr(
            service,
            "send",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "Notification service does not provide send()."
            )

        return method(
            message=message,
            title=title,
            **action.get(
                "data",
                {},
            ),
        )

    # ========================================================
    # VOICE
    # ========================================================

    def _speak(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:

        text = self._resolve(
            action.get(
                "text",
                action.get(
                    "message",
                    "",
                ),
            ),
            context,
        )

        if not text:
            raise ValueError(
                "Speak action requires text."
            )

        service = self.services.get(
            "voice"
        )

        if service is None:
            logger.info(
                "[RENIX VOICE] %s",
                text,
            )

            return {
                "status": "logged",
                "text": text,
            }

        method = getattr(
            service,
            "speak",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "Voice service does not provide speak()."
            )

        return method(
            text,
            **action.get(
                "options",
                {},
            ),
        )

    # ========================================================
    # BROWSER
    # ========================================================

    def _open_url(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        url = self._resolve(
            action.get(
                "url",
                "",
            ),
            context,
        )

        if not url:
            raise ValueError(
                "URL is required."
            )

        webbrowser.open(
            url
        )

        return {
            "status": "completed",
            "url": url,
        }

    def _browser(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:

        service = self.get_service(
            "browser"
        )

        method_name = action.get(
            "method",
            "execute",
        )

        method = getattr(
            service,
            method_name,
            None,
        )

        if not callable(method):
            raise RuntimeError(
                f"Browser service does not provide "
                f"{method_name}()."
            )

        return method(
            **action.get(
                "data",
                {},
            )
        )

    # ========================================================
    # COMPUTER
    # ========================================================

    def _computer(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:

        service = self.get_service(
            "computer"
        )

        method_name = action.get(
            "method",
            "execute",
        )

        method = getattr(
            service,
            method_name,
            None,
        )

        if not callable(method):
            raise RuntimeError(
                f"Computer service does not provide "
                f"{method_name}()."
            )

        return method(
            **action.get(
                "data",
                {},
            )
        )

    # ========================================================
    # FILES
    # ========================================================

    def _file(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:

        service = self.get_service(
            "files"
        )

        method_name = action.get(
            "method",
            "execute",
        )

        method = getattr(
            service,
            method_name,
            None,
        )

        if not callable(method):
            raise RuntimeError(
                f"File service does not provide "
                f"{method_name}()."
            )

        return method(
            **action.get(
                "data",
                {},
            )
        )

    # ========================================================
    # SMART HOME
    # ========================================================

    def _smart_home(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:

        service = self.get_service(
            "smart_home"
        )

        method_name = action.get(
            "method",
            "execute",
        )

        method = getattr(
            service,
            method_name,
            None,
        )

        if not callable(method):
            raise RuntimeError(
                f"Smart-home service does not provide "
                f"{method_name}()."
            )

        return method(
            **action.get(
                "data",
                {},
            )
        )

    # ========================================================
    # MEDIA
    # ========================================================

    def _media(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:

        service = self.get_service(
            "media"
        )

        method_name = action.get(
            "method",
            "execute",
        )

        method = getattr(
            service,
            method_name,
            None,
        )

        if not callable(method):
            raise RuntimeError(
                f"Media service does not provide "
                f"{method_name}()."
            )

        return method(
            **action.get(
                "data",
                {},
            )
        )

    # ========================================================
    # PERSONAL TASKS
    # ========================================================

    def _task(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:

        service = self.get_service(
            "tasks"
        )

        method_name = action.get(
            "method",
            "create",
        )

        method = getattr(
            service,
            method_name,
            None,
        )

        if not callable(method):
            raise RuntimeError(
                f"Task service does not provide "
                f"{method_name}()."
            )

        return method(
            **action.get(
                "data",
                {},
            )
        )

    # ========================================================
    # CALENDAR
    # ========================================================

    def _calendar(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:

        service = self.get_service(
            "calendar"
        )

        method_name = action.get(
            "method",
            "create_event",
        )

        method = getattr(
            service,
            method_name,
            None,
        )

        if not callable(method):
            raise RuntimeError(
                f"Calendar service does not provide "
                f"{method_name}()."
            )

        return method(
            **action.get(
                "data",
                {},
            )
        )

    # ========================================================
    # EMAIL
    # ========================================================

    def _email(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:

        service = self.get_service(
            "email"
        )

        method_name = action.get(
            "method",
            "send",
        )

        method = getattr(
            service,
            method_name,
            None,
        )

        if not callable(method):
            raise RuntimeError(
                f"Email service does not provide "
                f"{method_name}()."
            )

        return method(
            **action.get(
                "data",
                {},
            )
        )

    # ========================================================
    # CODING
    # ========================================================

    def _coding(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:

        service = self.get_service(
            "coding"
        )

        method_name = action.get(
            "method",
            "execute",
        )

        method = getattr(
            service,
            method_name,
            None,
        )

        if not callable(method):
            raise RuntimeError(
                f"Coding service does not provide "
                f"{method_name}()."
            )

        return method(
            **action.get(
                "data",
                {},
            )
        )

    # ========================================================
    # SYSTEM
    # ========================================================

    def _system(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:

        service = self.get_service(
            "system"
        )

        method_name = action.get(
            "method",
            "execute",
        )

        method = getattr(
            service,
            method_name,
            None,
        )

        if not callable(method):
            raise RuntimeError(
                f"System service does not provide "
                f"{method_name}()."
            )

        return method(
            **action.get(
                "data",
                {},
            )
        )

    # ========================================================
    # COMMAND EXECUTION
    # ========================================================

    def _run_command(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        command = action.get(
            "command"
        )

        if not command:
            raise ValueError(
                "Command is required."
            )

        if isinstance(
            command,
            str,
        ):
            shell = True
        else:
            shell = False

        completed = subprocess.run(
            command,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=float(
                action.get(
                    "timeout",
                    30,
                )
            ),
            check=False,
        )

        return {
            "status": (
                "completed"
                if completed.returncode == 0
                else "failed"
            ),
            "return_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    # ========================================================
    # RESOLUTION
    # ========================================================

    @staticmethod
    def _resolve(
        value: Any,
        context: dict[str, Any],
    ) -> Any:
        """
        Resolve simple {{variable}} expressions.
        """

        if not isinstance(
            value,
            str,
        ):
            return value

        result = value

        for key, replacement in context.items():

            token = "{{" + str(key) + "}}"

            result = result.replace(
                token,
                str(replacement),
            )

        return result

    # ========================================================
    # SECURITY
    # ========================================================

    def _check_security(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> None:

        if self.security_manager is None:
            return

        checker = getattr(
            self.security_manager,
            "authorize_action",
            None,
        )

        if callable(checker):

            allowed = checker(
                action,
                context=context,
            )

            if allowed is False:
                raise PermissionError(
                    "Automation action was denied by security manager."
                )

    # ========================================================
    # EVENTS
    # ========================================================

    def _emit(
        self,
        event: str,
        payload: dict[str, Any],
    ) -> None:

        if self.event_bus is None:
            return

        publish = getattr(
            self.event_bus,
            "publish",
            None,
        )

        if callable(publish):

            try:
                publish(
                    event,
                    payload,
                )
            except Exception:
                logger.exception(
                    "Failed to publish automation event."
                )


__all__ = [
    "ActionExecutor",
]


