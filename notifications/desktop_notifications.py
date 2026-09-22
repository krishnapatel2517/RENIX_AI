"""
RENIX Notifications - Desktop Notifications
============================================

Desktop notification backend for RENIX.

This module provides a small platform-aware abstraction for
displaying native desktop notifications.

Supported backends:
- Windows Toast notifications through winrt when available
- Windows fallback through PowerShell
- Linux desktop notifications through notify-send
- macOS notifications through osascript

The module does not make the notification manager dependent
on any specific operating system.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Mapping

from .notification_manager import (
    Notification,
    NotificationManager,
)


@dataclass
class DesktopNotificationResult:
    """Result returned after attempting a desktop notification."""

    success: bool
    backend: str
    message: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "backend": self.backend,
            "message": self.message,
            "error": self.error,
        }


class DesktopNotificationManager:
    """
    Handles desktop notifications for RENIX.

    The class can be used independently:

        desktop = DesktopNotificationManager()
        desktop.show("RENIX", "System ready.")

    Or attached to NotificationManager:

        manager = NotificationManager()
        desktop.attach(manager)
    """

    def __init__(
        self,
        *,
        app_name: str = "RENIX",
        enabled: bool = True,
        async_mode: bool = False,
    ) -> None:
        self.app_name = app_name
        self.enabled = enabled
        self.async_mode = async_mode

        self._manager: NotificationManager | None = None
        self._handler_name = (
            "desktop_notification_manager"
        )

        self._lock = threading.RLock()

        self._last_result = DesktopNotificationResult(
            success=False,
            backend="none",
            message="No notification has been sent yet.",
        )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    def enable(self) -> None:
        """Enable desktop notifications."""

        with self._lock:
            self.enabled = True

    def disable(self) -> None:
        """Disable desktop notifications."""

        with self._lock:
            self.enabled = False

    def is_enabled(self) -> bool:
        """Return whether desktop notifications are enabled."""

        with self._lock:
            return self.enabled

    # ========================================================
    # PLATFORM
    # ========================================================

    @staticmethod
    def platform_name() -> str:
        """Return the current operating system."""

        system = platform.system().lower()

        if system == "windows":
            return "windows"

        if system == "darwin":
            return "macos"

        if system == "linux":
            return "linux"

        return system or "unknown"

    def available_backends(self) -> list[str]:
        """Return notification backends available on this machine."""

        system = self.platform_name()
        backends: list[str] = []

        if system == "windows":
            if self._winrt_available():
                backends.append("windows_toast")

            if shutil.which("powershell"):
                backends.append("powershell")

        elif system == "linux":
            if shutil.which("notify-send"):
                backends.append("notify-send")

        elif system == "macos":
            if shutil.which("osascript"):
                backends.append("osascript")

        return backends

    def preferred_backend(self) -> str:
        """Return the best available backend."""

        backends = self.available_backends()

        if not backends:
            return "none"

        return backends[0]

    # ========================================================
    # SHOW
    # ========================================================

    def show(
        self,
        title: str,
        message: str,
        *,
        notification: Notification | None = None,
        timeout: int | None = None,
        icon: str | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> DesktopNotificationResult:
        """
        Display a desktop notification.

        Returns a DesktopNotificationResult rather than raising
        platform-specific notification errors.
        """

        if not self.enabled:
            result = DesktopNotificationResult(
                success=False,
                backend="disabled",
                message="Desktop notifications are disabled.",
            )

            self._set_last_result(result)
            return result

        title = str(title)
        message = str(message)

        if self.async_mode:
            thread = threading.Thread(
                target=self._show_sync,
                args=(
                    title,
                    message,
                    notification,
                    timeout,
                    icon,
                    data,
                ),
                daemon=True,
            )

            thread.start()

            result = DesktopNotificationResult(
                success=True,
                backend="async",
                message="Notification dispatch started.",
            )

            self._set_last_result(result)
            return result

        return self._show_sync(
            title,
            message,
            notification,
            timeout,
            icon,
            data,
        )

    def _show_sync(
        self,
        title: str,
        message: str,
        notification: Notification | None,
        timeout: int | None,
        icon: str | None,
        data: Mapping[str, Any] | None,
    ) -> DesktopNotificationResult:
        backend = self.preferred_backend()

        try:
            if backend == "windows_toast":
                result = self._show_windows_toast(
                    title,
                    message,
                    notification=notification,
                    icon=icon,
                )

            elif backend == "powershell":
                result = self._show_windows_powershell(
                    title,
                    message,
                )

            elif backend == "notify-send":
                result = self._show_linux(
                    title,
                    message,
                    notification=notification,
                    timeout=timeout,
                    icon=icon,
                )

            elif backend == "osascript":
                result = self._show_macos(
                    title,
                    message,
                )

            else:
                result = DesktopNotificationResult(
                    success=False,
                    backend="none",
                    message="No supported desktop notification backend found.",
                )

        except Exception as exc:
            result = DesktopNotificationResult(
                success=False,
                backend=backend,
                message="Desktop notification failed.",
                error=str(exc),
            )

        self._set_last_result(result)

        return result

    # ========================================================
    # WINDOWS TOAST
    # ========================================================

    @staticmethod
    def _winrt_available() -> bool:
        """Check whether WinRT is installed."""

        try:
            import winrt.windows.ui.notifications  # noqa: F401
            import winrt.windows.data.xml.dom  # noqa: F401

            return True

        except ImportError:
            return False

    def _show_windows_toast(
        self,
        title: str,
        message: str,
        *,
        notification: Notification | None = None,
        icon: str | None = None,
    ) -> DesktopNotificationResult:
        """
        Display a Windows Toast notification.

        Requires the optional `winrt` package.
        """

        try:
            from winrt.windows.data.xml.dom import (
                XmlDocument,
            )
            from winrt.windows.ui.notifications import (
                ToastNotification,
                ToastNotificationManager,
            )

        except ImportError as exc:
            raise RuntimeError(
                "Windows Toast backend requires the "
                "'winrt' package."
            ) from exc

        safe_title = self._xml_escape(title)
        safe_message = self._xml_escape(message)

        xml = (
            "<toast>"
            "<visual>"
            "<binding template='ToastGeneric'>"
            f"<text>{safe_title}</text>"
            f"<text>{safe_message}</text>"
            "</binding>"
            "</visual>"
            "</toast>"
        )

        document = XmlDocument()
        document.load_xml(xml)

        toast = ToastNotification(
            document
        )

        notifier = (
            ToastNotificationManager
            .create_toast_notifier(
                self.app_name
            )
        )

        notifier.show(toast)

        return DesktopNotificationResult(
            success=True,
            backend="windows_toast",
            message="Windows Toast notification displayed.",
        )

    # ========================================================
    # WINDOWS POWERSHELL FALLBACK
    # ========================================================

    def _show_windows_powershell(
        self,
        title: str,
        message: str,
    ) -> DesktopNotificationResult:
        """
        Windows fallback using PowerShell.

        This uses a simple Windows Forms balloon notification.
        """

        if not shutil.which("powershell"):
            return DesktopNotificationResult(
                success=False,
                backend="powershell",
                message="PowerShell is not available.",
            )

        encoded_title = self._powershell_escape(
            title
        )
        encoded_message = self._powershell_escape(
            message
        )

        script = f"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$notify = New-Object System.Windows.Forms.NotifyIcon
$notify.Icon = [System.Drawing.SystemIcons]::Information
$notify.Visible = $true
$notify.BalloonTipTitle = '{encoded_title}'
$notify.BalloonTipText = '{encoded_message}'
$notify.ShowBalloonTip(5000)

Start-Sleep -Seconds 6

$notify.Dispose()
"""

        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                completed.stderr.strip()
                or "PowerShell notification failed."
            )

        return DesktopNotificationResult(
            success=True,
            backend="powershell",
            message="Windows notification displayed.",
        )

    # ========================================================
    # LINUX
    # ========================================================

    def _show_linux(
        self,
        title: str,
        message: str,
        *,
        notification: Notification | None = None,
        timeout: int | None = None,
        icon: str | None = None,
    ) -> DesktopNotificationResult:
        """Display a Linux desktop notification."""

        command = [
            "notify-send",
            title,
            message,
        ]

        if icon:
            command.extend(
                [
                    "-i",
                    icon,
                ]
            )

        if timeout is not None:
            timeout_ms = max(
                0,
                int(timeout * 1000),
            )

            command.extend(
                [
                    "-t",
                    str(timeout_ms),
                ]
            )

        if notification is not None:
            urgency = self._linux_urgency(
                notification
            )

            command.extend(
                [
                    "-u",
                    urgency,
                ]
            )

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                completed.stderr.strip()
                or "notify-send failed."
            )

        return DesktopNotificationResult(
            success=True,
            backend="notify-send",
            message="Linux desktop notification displayed.",
        )

    @staticmethod
    def _linux_urgency(
        notification: Notification,
    ) -> str:
        """Map RENIX notification priority to Linux urgency."""

        priority_name = (
            notification.priority.name.lower()
        )

        if priority_name in {
            "critical",
            "urgent",
        }:
            return "critical"

        if priority_name in {
            "high",
            "warning",
        }:
            return "normal"

        return "low"

    # ========================================================
    # MACOS
    # ========================================================

    def _show_macos(
        self,
        title: str,
        message: str,
    ) -> DesktopNotificationResult:
        """Display a macOS notification using osascript."""

        script = (
            "display notification "
            f"{self._applescript_string(message)} "
            "with title "
            f"{self._applescript_string(title)}"
        )

        completed = subprocess.run(
            [
                "osascript",
                "-e",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                completed.stderr.strip()
                or "osascript notification failed."
            )

        return DesktopNotificationResult(
            success=True,
            backend="osascript",
            message="macOS notification displayed.",
        )

    # ========================================================
    # NOTIFICATION MANAGER INTEGRATION
    # ========================================================

    def attach(
        self,
        manager: NotificationManager,
        *,
        category: str = "*",
    ) -> None:
        """
        Attach this desktop backend to NotificationManager.
        """

        if self._manager is not None:
            self.detach()

        self._manager = manager

        manager.register_handler(
            self._handler_name,
            self._handle_notification,
            category=category,
        )

    def detach(
        self,
    ) -> None:
        """Detach from the current NotificationManager."""

        manager = self._manager

        if manager is None:
            return

        manager.unregister_handler(
            self._handler_name,
            category="*",
        )

        self._manager = None

    def _handle_notification(
        self,
        notification: Notification,
    ) -> bool:
        """Handle a RENIX notification."""

        result = self.show(
            notification.title,
            notification.message,
            notification=notification,
            data=notification.data,
        )

        return result.success

    # ========================================================
    # LAST RESULT
    # ========================================================

    def last_result(
        self,
    ) -> DesktopNotificationResult:
        """Return the most recent notification result."""

        with self._lock:
            return self._last_result

    def _set_last_result(
        self,
        result: DesktopNotificationResult,
    ) -> None:
        with self._lock:
            self._last_result = result

    # ========================================================
    # UTILITY
    # ========================================================

    @staticmethod
    def _xml_escape(
        value: str,
    ) -> str:
        return (
            value
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    @staticmethod
    def _powershell_escape(
        value: str,
    ) -> str:
        """
        Escape text for a PowerShell single-quoted string.
        """

        return str(value).replace(
            "'",
            "''",
        )

    @staticmethod
    def _applescript_string(
        value: str,
    ) -> str:
        """
        Convert Python text into a safe AppleScript string.
        """

        escaped = (
            str(value)
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )

        return f'"{escaped}"'

    def status(self) -> dict[str, Any]:
        """Return desktop notification backend status."""

        return {
            "enabled": self.enabled,
            "platform": self.platform_name(),
            "preferred_backend": self.preferred_backend(),
            "available_backends": self.available_backends(),
            "async_mode": self.async_mode,
            "attached": self._manager is not None,
            "last_result": (
                self.last_result().to_dict()
            ),
        }

    def __enter__(
        self,
    ) -> "DesktopNotificationManager":
        self.enable()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.detach()

    def __repr__(self) -> str:
        return (
            "DesktopNotificationManager("
            f"app_name={self.app_name!r}, "
            f"enabled={self.enabled}, "
            f"platform={self.platform_name()!r})"
        )


__all__ = [
    "DesktopNotificationManager",
    "DesktopNotificationResult",
]


