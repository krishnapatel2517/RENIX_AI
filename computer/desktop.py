"""
RENIX Desktop Controller
========================

Desktop-level operations for RENIX.

Features:
- Show desktop
- Minimize all windows
- Restore windows
- Open common desktop locations
- Refresh desktop
- Get desktop paths
- Create desktop shortcuts
- Launch desktop/explorer
- Lock workstation
- Basic desktop state

Windows-focused implementation.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DesktopController:
    """
    High-level Windows desktop controller for RENIX.

    Example:

        desktop = DesktopController()

        desktop.show_desktop()
        desktop.open_desktop_folder()
        desktop.open_downloads()
        desktop.refresh()
    """

    def __init__(
        self,
        enabled: bool = True,
    ) -> None:

        self.enabled = enabled

        self._shell = None

        logger.info(
            "RENIX desktop controller initialized."
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
                "RENIX desktop control is disabled."
            )

    # ==========================================================
    # PATHS
    # ==========================================================

    def get_desktop_path(self) -> Path:

        return Path(
            os.path.join(
                os.path.expanduser("~"),
                "Desktop",
            )
        )

    def get_documents_path(self) -> Path:

        return Path(
            os.path.join(
                os.path.expanduser("~"),
                "Documents",
            )
        )

    def get_downloads_path(self) -> Path:

        return Path(
            os.path.join(
                os.path.expanduser("~"),
                "Downloads",
            )
        )

    def get_pictures_path(self) -> Path:

        return Path(
            os.path.join(
                os.path.expanduser("~"),
                "Pictures",
            )
        )

    def get_videos_path(self) -> Path:

        return Path(
            os.path.join(
                os.path.expanduser("~"),
                "Videos",
            )
        )

    def get_music_path(self) -> Path:

        return Path(
            os.path.join(
                os.path.expanduser("~"),
                "Music",
            )
        )

    def get_home_path(self) -> Path:

        return Path(
            os.path.expanduser("~")
        )

    def get_paths(self) -> dict[str, str]:

        return {
            "home": str(
                self.get_home_path()
            ),
            "desktop": str(
                self.get_desktop_path()
            ),
            "documents": str(
                self.get_documents_path()
            ),
            "downloads": str(
                self.get_downloads_path()
            ),
            "pictures": str(
                self.get_pictures_path()
            ),
            "videos": str(
                self.get_videos_path()
            ),
            "music": str(
                self.get_music_path()
            ),
        }

    # ==========================================================
    # OPEN FOLDERS
    # ==========================================================

    def open_folder(
        self,
        path: str | Path,
    ) -> bool:

        self._check()

        folder = (
            Path(path)
            .expanduser()
            .resolve()
        )

        if not folder.exists():
            raise FileNotFoundError(
                f"Folder does not exist: {folder}"
            )

        if not folder.is_dir():
            raise NotADirectoryError(
                f"Not a directory: {folder}"
            )

        try:

            subprocess.Popen(
                [
                    "explorer.exe",
                    str(folder),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return True

        except Exception as exc:

            logger.exception(
                "Unable to open folder."
            )

            raise RuntimeError(
                f"Unable to open folder: {exc}"
            ) from exc

    def open_desktop_folder(self) -> bool:

        return self.open_folder(
            self.get_desktop_path()
        )

    def open_documents(self) -> bool:

        return self.open_folder(
            self.get_documents_path()
        )

    def open_downloads(self) -> bool:

        return self.open_folder(
            self.get_downloads_path()
        )

    def open_pictures(self) -> bool:

        return self.open_folder(
            self.get_pictures_path()
        )

    def open_videos(self) -> bool:

        return self.open_folder(
            self.get_videos_path()
        )

    def open_music(self) -> bool:

        return self.open_folder(
            self.get_music_path()
        )

    def open_home(self) -> bool:

        return self.open_folder(
            self.get_home_path()
        )

    # ==========================================================
    # SHOW DESKTOP
    # ==========================================================

    def show_desktop(self) -> bool:
        """
        Minimize all currently visible windows.
        """

        self._check()

        try:

            import ctypes

            shell = ctypes.windll.user32

            # Windows desktop shortcut:
            # Win + D
            shell.keybd_event(
                0x5B,
                0,
                0,
                0,
            )

            shell.keybd_event(
                0x44,
                0,
                0,
                0,
            )

            shell.keybd_event(
                0x44,
                0,
                2,
                0,
            )

            shell.keybd_event(
                0x5B,
                0,
                2,
                0,
            )

            return True

        except Exception as exc:

            logger.warning(
                "Direct desktop command failed: %s",
                exc,
            )

        # Fallback to PowerShell.
        try:

            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        "$shell = "
                        "New-Object -ComObject "
                        "Shell.Application; "
                        "$shell.MinimizeAll()"
                    ),
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return True

        except Exception:

            return False

    def minimize_all(self) -> bool:

        return self.show_desktop()

    # ==========================================================
    # RESTORE WINDOWS
    # ==========================================================

    def restore_windows(self) -> bool:

        self._check()

        try:

            import ctypes

            user32 = ctypes.windll.user32

            # Win + D toggles the desktop state.
            user32.keybd_event(
                0x5B,
                0,
                0,
                0,
            )

            user32.keybd_event(
                0x44,
                0,
                0,
                0,
            )

            user32.keybd_event(
                0x44,
                0,
                2,
                0,
            )

            user32.keybd_event(
                0x5B,
                0,
                2,
                0,
            )

            return True

        except Exception:

            return False

    # ==========================================================
    # REFRESH
    # ==========================================================

    def refresh(self) -> bool:
        """
        Refresh the Windows desktop / Explorer.
        """

        self._check()

        try:

            subprocess.run(
                [
                    "ie4uinit.exe",
                    "-show",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return True

        except Exception as exc:

            logger.warning(
                "Desktop refresh failed: %s",
                exc,
            )

            return False

    # ==========================================================
    # OPEN DESKTOP
    # ==========================================================

    def open_explorer(self) -> bool:

        self._check()

        try:

            subprocess.Popen(
                ["explorer.exe"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return True

        except Exception:

            return False

    # ==========================================================
    # CREATE DESKTOP FOLDER
    # ==========================================================

    def create_desktop_folder(
        self,
        name: str,
    ) -> Path:

        self._check()

        name = str(name).strip()

        if not name:
            raise ValueError(
                "Folder name cannot be empty."
            )

        desktop = (
            self.get_desktop_path()
        )

        desktop.mkdir(
            parents=True,
            exist_ok=True,
        )

        folder = desktop / name

        folder.mkdir(
            parents=True,
            exist_ok=False,
        )

        logger.info(
            "Created desktop folder: %s",
            folder,
        )

        return folder

    # ==========================================================
    # CREATE SHORTCUT
    # ==========================================================

    def create_shortcut(
        self,
        name: str,
        target: str,
        description: str = "",
        working_directory: Optional[str] = None,
        icon_path: Optional[str] = None,
    ) -> Path:
        """
        Create a Windows .lnk shortcut on the desktop.
        """

        self._check()

        name = str(name).strip()

        if not name:
            raise ValueError(
                "Shortcut name cannot be empty."
            )

        target = str(
            Path(target)
            .expanduser()
            .resolve()
        )

        desktop = (
            self.get_desktop_path()
        )

        desktop.mkdir(
            parents=True,
            exist_ok=True,
        )

        shortcut_path = (
            desktop
            / f"{name}.lnk"
        )

        # PowerShell COM API creates native
        # Windows shortcuts without requiring
        # an additional Python package.
        shortcut_value = self._escape_powershell(str(shortcut_path))
        target_value = self._escape_powershell(target)
        script = (
            "$ws = "
            "New-Object -ComObject "
            "WScript.Shell; "
            f"$s = $ws.CreateShortcut('{shortcut_value}'); "
            f"$s.TargetPath = "
            f"'{target_value}'; "
        )

        if description:
            description_value = self._escape_powershell(description)

            script += (
                f"$s.Description = "
                f"'{description_value}'; "
            )

        if working_directory:
            working_directory_value = self._escape_powershell(
                str(
                    Path(
                        working_directory
                    )
                    .expanduser()
                    .resolve()
                )
            )

            script += (
                f"$s.WorkingDirectory = "
                f"'{working_directory_value}'; "
            )

        if icon_path:
            icon_value = self._escape_powershell(
                str(
                    Path(icon_path)
                    .expanduser()
                    .resolve()
                )
            )

            script += (
                f"$s.IconLocation = "
                f"'{icon_value}'; "
            )

        script += "$s.Save()"

        try:

            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    script,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return shortcut_path

        except Exception as exc:

            raise RuntimeError(
                f"Unable to create shortcut: {exc}"
            ) from exc

    @staticmethod
    def _escape_powershell(
        value: str,
    ) -> str:

        return str(value).replace(
            "'",
            "''",
        )

    # ==========================================================
    # DELETE SHORTCUT
    # ==========================================================

    def delete_shortcut(
        self,
        name: str,
    ) -> bool:

        self._check()

        name = str(name).strip()

        path = (
            self.get_desktop_path()
            / (
                name
                if name.lower().endswith(".lnk")
                else f"{name}.lnk"
            )
        )

        if not path.exists():
            return False

        try:

            path.unlink()

            return True

        except Exception as exc:

            logger.warning(
                "Unable to delete shortcut: %s",
                exc,
            )

            return False

    # ==========================================================
    # DESKTOP ITEMS
    # ==========================================================

    def list_desktop_items(
        self,
        include_hidden: bool = False,
    ) -> list[dict[str, Any]]:

        self._check()

        desktop = (
            self.get_desktop_path()
        )

        if not desktop.exists():
            return []

        results = []

        for item in desktop.iterdir():

            if (
                not include_hidden
                and item.name.startswith(".")
            ):
                continue

            try:

                stat = item.stat()

                results.append(
                    {
                        "name": item.name,
                        "path": str(item),
                        "is_file": item.is_file(),
                        "is_directory": item.is_dir(),
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    }
                )

            except OSError:

                continue

        results.sort(
            key=lambda item: item["name"].lower()
        )

        return results

    # ==========================================================
    # OPEN RECYCLE BIN
    # ==========================================================

    def open_recycle_bin(self) -> bool:

        self._check()

        try:

            subprocess.Popen(
                [
                    "explorer.exe",
                    "shell:RecycleBinFolder",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return True

        except Exception:

            return False

    # ==========================================================
    # LOCK COMPUTER
    # ==========================================================

    def lock(self) -> bool:
        """
        Lock the Windows workstation.

        This operation immediately locks the session.
        """

        self._check()

        try:

            import ctypes

            ctypes.windll.user32.LockWorkStation()

            return True

        except Exception as exc:

            logger.warning(
                "Unable to lock workstation: %s",
                exc,
            )

            return False

    # ==========================================================
    # SLEEP
    # ==========================================================

    def sleep(self) -> bool:
        """
        Put Windows into sleep mode.
        """

        self._check()

        try:

            subprocess.run(
                [
                    "rundll32.exe",
                    "powrprof.dll,SetSuspendState",
                    "0,1,0",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return True

        except Exception as exc:

            logger.warning(
                "Unable to sleep computer: %s",
                exc,
            )

            return False

    # ==========================================================
    # DESKTOP EXISTENCE
    # ==========================================================

    def desktop_exists(self) -> bool:

        return self.get_desktop_path().exists()

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        desktop = (
            self.get_desktop_path()
        )

        return {
            "enabled": self.enabled,
            "platform": os.name,
            "desktop_path": str(
                desktop
            ),
            "desktop_exists": (
                desktop.exists()
            ),
            "desktop_items": (
                len(
                    self.list_desktop_items()
                )
                if self.enabled
                else 0
            ),
        }

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        self._shell = None

        logger.info(
            "RENIX desktop controller shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================

_default_desktop: Optional[
    DesktopController
] = None


def get_desktop_controller() -> DesktopController:
    """Return the shared RENIX desktop controller."""

    global _default_desktop

    if _default_desktop is None:

        _default_desktop = (
            DesktopController()
        )

    return _default_desktop


DesktopManager = DesktopController


