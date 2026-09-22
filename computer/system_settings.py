"""
RENIX System Settings Controller
================================

Windows system-level settings and utilities for RENIX.

Features:
- Open Windows Settings
- Open specific Settings pages
- Open Control Panel
- Open Task Manager
- Open Device Manager
- Open Services
- Open Network Settings
- Open Display Settings
- Open Sound Settings
- Open Bluetooth Settings
- Open Windows Update
- Open Privacy Settings
- Open Apps Settings
- Open Personalization
- Open Power Settings
- Run Windows system commands
- Query basic system information

Windows 10/11 focused.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SystemSettingsController:
    """
    High-level Windows settings controller for RENIX.

    Example:

        settings = SystemSettingsController()

        settings.open_settings()
        settings.open_display_settings()
        settings.open_sound_settings()
    """

    def __init__(
        self,
        enabled: bool = True,
    ) -> None:

        self.enabled = enabled

        logger.info(
            "RENIX system settings controller initialized."
        )

    # ==========================================================
    # STATE
    # ==========================================================

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def is_enabled(self) -> bool:
        return self.enabled

    def _check(self) -> None:

        if not self.enabled:
            raise RuntimeError(
                "RENIX system settings control is disabled."
            )

        if os.name != "nt":
            raise RuntimeError(
                "SystemSettingsController currently "
                "supports Windows only."
            )

    # ==========================================================
    # INTERNAL LAUNCHER
    # ==========================================================

    def _run(
        self,
        command: list[str],
        wait: bool = False,
    ) -> bool:

        self._check()

        try:

            if wait:

                subprocess.run(
                    command,
                    check=False,
                )

            else:

                subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            return True

        except Exception as exc:

            logger.warning(
                "Unable to execute system command %s: %s",
                command,
                exc,
            )

            return False

    # ==========================================================
    # WINDOWS SETTINGS
    # ==========================================================

    def open_settings(
        self,
        uri: Optional[str] = None,
    ) -> bool:
        """
        Open Windows Settings.

        If uri is supplied, it may be a Windows
        ms-settings URI.
        """

        self._check()

        target = (
            "ms-settings:"
            if uri is None
            else str(uri)
        )

        if not target.startswith(
            "ms-settings:"
        ):
            target = (
                "ms-settings:"
                + target
            )

        return self._run(
            [
                "explorer.exe",
                target,
            ]
        )

    # ==========================================================
    # COMMON SETTINGS
    # ==========================================================

    def open_system_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:system"
        )

    def open_display_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:display"
        )

    def open_sound_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:sound"
        )

    def open_notifications_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:notifications"
        )

    def open_focus_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:quiethours"
        )

    def open_power_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:powersleep"
        )

    def open_battery_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:batterysaver"
        )

    def open_storage_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:storagesense"
        )

    def open_multitasking_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:multitasking"
        )

    def open_remote_desktop_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:remotedesktop"
        )

    # ==========================================================
    # NETWORK
    # ==========================================================

    def open_network_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:network"
        )

    def open_wifi_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:network-wifi"
        )

    def open_ethernet_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:network-ethernet"
        )

    def open_vpn_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:network-vpn"
        )

    def open_proxy_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:network-proxy"
        )

    def open_hotspot_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:network-mobilehotspot"
        )

    # ==========================================================
    # BLUETOOTH / DEVICES
    # ==========================================================

    def open_bluetooth_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:bluetooth"
        )

    def open_devices_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:bluetooth"
        )

    def open_mouse_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:mousetouchpad"
        )

    def open_touchpad_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:devices-touchpad"
        )

    def open_typing_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:typing"
        )

    # ==========================================================
    # PERSONALIZATION
    # ==========================================================

    def open_personalization(self) -> bool:

        return self.open_settings(
            "ms-settings:personalization"
        )

    def open_background_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:personalization-background"
        )

    def open_colors_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:colors"
        )

    def open_themes_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:themes"
        )

    def open_lock_screen_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:lockscreen"
        )

    def open_taskbar_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:taskbar"
        )

    # ==========================================================
    # ACCOUNTS
    # ==========================================================

    def open_accounts(self) -> bool:

        return self.open_settings(
            "ms-settings:accounts"
        )

    def open_signin_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:signinoptions"
        )

    def open_family_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:family-group"
        )

    # ==========================================================
    # APPS
    # ==========================================================

    def open_apps_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:appsfeatures"
        )

    def open_default_apps(self) -> bool:

        return self.open_settings(
            "ms-settings:defaultapps"
        )

    def open_startup_apps(self) -> bool:

        return self.open_settings(
            "ms-settings:startupapps"
        )

    # ==========================================================
    # PRIVACY
    # ==========================================================

    def open_privacy_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:privacy"
        )

    def open_camera_privacy(self) -> bool:

        return self.open_settings(
            "ms-settings:privacy-webcam"
        )

    def open_microphone_privacy(self) -> bool:

        return self.open_settings(
            "ms-settings:privacy-microphone"
        )

    def open_location_privacy(self) -> bool:

        return self.open_settings(
            "ms-settings:privacy-location"
        )

    # ==========================================================
    # WINDOWS UPDATE
    # ==========================================================

    def open_windows_update(self) -> bool:

        return self.open_settings(
            "ms-settings:windowsupdate"
        )

    def open_windows_update_history(self) -> bool:

        return self.open_settings(
            "ms-settings:windowsupdate-history"
        )

    def open_advanced_update_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:windowsupdate-options"
        )

    # ==========================================================
    # SEARCH
    # ==========================================================

    def open_search_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:search"
        )

    # ==========================================================
    # TIME / LANGUAGE
    # ==========================================================

    def open_time_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:dateandtime"
        )

    def open_language_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:regionlanguage"
        )

    def open_typing_language_settings(self) -> bool:

        return self.open_settings(
            "ms-settings:typing"
        )

    # ==========================================================
    # ACCESSIBILITY
    # ==========================================================

    def open_accessibility(self) -> bool:

        return self.open_settings(
            "ms-settings:easeofaccess"
        )

    def open_display_accessibility(self) -> bool:

        return self.open_settings(
            "ms-settings:easeofaccess-display"
        )

    def open_keyboard_accessibility(self) -> bool:

        return self.open_settings(
            "ms-settings:easeofaccess-keyboard"
        )

    def open_mouse_accessibility(self) -> bool:

        return self.open_settings(
            "ms-settings:easeofaccess-mouse"
        )

    # ==========================================================
    # CONTROL PANEL
    # ==========================================================

    def open_control_panel(self) -> bool:

        return self._run(
            [
                "control.exe",
            ]
        )

    def open_network_connections(self) -> bool:

        return self._run(
            [
                "control.exe",
                "ncpa.cpl",
            ]
        )

    def open_programs_features(self) -> bool:

        return self._run(
            [
                "control.exe",
                "appwiz.cpl",
            ]
        )

    def open_sound_control_panel(self) -> bool:

        return self._run(
            [
                "control.exe",
                "mmsys.cpl",
            ]
        )

    def open_mouse_control_panel(self) -> bool:

        return self._run(
            [
                "control.exe",
                "main.cpl",
            ]
        )

    # ==========================================================
    # ADMIN / SYSTEM UTILITIES
    # ==========================================================

    def open_task_manager(self) -> bool:

        return self._run(
            [
                "taskmgr.exe",
            ]
        )

    def open_device_manager(self) -> bool:

        return self._run(
            [
                "devmgmt.msc",
            ]
        )

    def open_services(self) -> bool:

        return self._run(
            [
                "services.msc",
            ]
        )

    def open_event_viewer(self) -> bool:

        return self._run(
            [
                "eventvwr.msc",
            ]
        )

    def open_disk_management(self) -> bool:

        return self._run(
            [
                "diskmgmt.msc",
            ]
        )

    def open_computer_management(self) -> bool:

        return self._run(
            [
                "compmgmt.msc",
            ]
        )

    def open_system_information(self) -> bool:

        return self._run(
            [
                "msinfo32.exe",
            ]
        )

    def open_resource_monitor(self) -> bool:

        return self._run(
            [
                "resmon.exe",
            ]
        )

    # ==========================================================
    # COMMAND PROMPT / POWERSHELL
    # ==========================================================

    def open_command_prompt(self) -> bool:

        return self._run(
            [
                "cmd.exe",
            ]
        )

    def open_powershell(self) -> bool:

        return self._run(
            [
                "powershell.exe",
                "-NoProfile",
            ]
        )

    def open_windows_terminal(self) -> bool:

        return self._run(
            [
                "wt.exe",
            ]
        )

    # ==========================================================
    # SYSTEM INFORMATION
    # ==========================================================

    def get_system_information(
        self,
    ) -> dict[str, Any]:

        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
        }

    # ==========================================================
    # ENVIRONMENT
    # ==========================================================

    def get_environment_variable(
        self,
        name: str,
        default: Optional[str] = None,
    ) -> Optional[str]:

        return os.environ.get(
            str(name),
            default,
        )

    def get_environment(
        self,
    ) -> dict[str, str]:

        return dict(os.environ)

    # ==========================================================
    # URI LAUNCHER
    # ==========================================================

    def open_uri(
        self,
        uri: str,
    ) -> bool:
        """
        Open a Windows URI.

        Examples:

            open_uri("ms-settings:")
            open_uri("https://example.com")
        """

        self._check()

        uri = str(uri).strip()

        if not uri:
            raise ValueError(
                "URI cannot be empty."
            )

        return self._run(
            [
                "explorer.exe",
                uri,
            ]
        )

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "platform": platform.system(),
            "windows": os.name == "nt",
        }

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX system settings controller shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================

_default_settings: Optional[
    SystemSettingsController
] = None


def get_system_settings() -> SystemSettingsController:
    """Return the shared RENIX system settings controller."""

    global _default_settings

    if _default_settings is None:

        _default_settings = (
            SystemSettingsController()
        )

    return _default_settings


SystemSettings = SystemSettingsController


