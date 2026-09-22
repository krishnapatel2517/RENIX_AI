"""
RENIX Browser Download Manager
==============================

High-level browser download management.

Responsibilities:
    - Download files from URLs
    - Track active downloads
    - Detect download completion
    - Manage download destinations
    - Prevent unsafe path traversal
    - Maintain download history
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)


@dataclass
class Download:
    """Represents a browser download."""

    id: str
    url: str
    filename: str
    path: str
    status: str = "pending"
    size: int = 0
    started_at: float = 0.0
    completed_at: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DownloadManager:
    """Manage browser downloads."""

    def __init__(
        self,
        browser_manager: Any = None,
        download_directory: str | os.PathLike[str] = "downloads",
    ) -> None:

        self.browser_manager = browser_manager

        self.download_directory = Path(
            download_directory
        ).expanduser().resolve()

        self.download_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._downloads: dict[
            str,
            Download,
        ] = {}

        self._lock = threading.RLock()

    # ========================================================
    # DIRECTORY
    # ========================================================

    def set_download_directory(
        self,
        directory: str | os.PathLike[str],
    ) -> Path:
        """Set and create the download directory."""

        path = Path(
            directory
        ).expanduser().resolve()

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.download_directory = path

        return path

    def get_download_directory(self) -> Path:
        return self.download_directory

    # ========================================================
    # DOWNLOAD
    # ========================================================

    def download(
        self,
        url: str,
        *,
        filename: str | None = None,
        page_id: str | None = None,
        timeout: float = 120.0,
    ) -> Download:
        """
        Download a URL using the configured browser.

        Supports browser backends exposing a download()
        method or Playwright-style expect_download().
        """

        if not url or not url.strip():
            raise ValueError(
                "Download URL cannot be empty."
            )

        url = url.strip()

        download_id = str(
            uuid.uuid4()
        )

        safe_filename = self._sanitize_filename(
            filename
            or self._filename_from_url(url)
        )

        destination = self._unique_path(
            self.download_directory / safe_filename
        )

        record = Download(
            id=download_id,
            url=url,
            filename=destination.name,
            path=str(destination),
            status="pending",
            started_at=time.time(),
        )

        with self._lock:
            self._downloads[
                download_id
            ] = record

        try:

            page = self._get_page(
                page_id
            )

            record.status = "downloading"

            result = self._perform_download(
                page,
                url,
                timeout,
            )

            self._save_result(
                result,
                destination,
            )

            record.status = "completed"
            record.completed_at = time.time()

            self._update_size(
                record
            )

            return record

        except Exception as exc:

            record.status = "failed"
            record.error = str(
                exc
            )
            record.completed_at = time.time()

            logger.exception(
                "Download failed: %s",
                url,
            )

            raise

    # ========================================================
    # BROWSER DOWNLOAD
    # ========================================================

    def _perform_download(
        self,
        page: Any,
        url: str,
        timeout: float,
    ) -> Any:

        # Direct browser-manager/page download support.
        direct_download = getattr(
            page,
            "download",
            None,
        )

        if callable(direct_download):

            try:
                return direct_download(
                    url,
                    timeout=timeout * 1000,
                )
            except TypeError:
                return direct_download(
                    url
                )

        # Playwright-style download through a URL.
        expect_download = getattr(
            page,
            "expect_download",
            None,
        )

        goto = getattr(
            page,
            "goto",
            None,
        )

        if callable(expect_download) and callable(goto):

            try:
                with expect_download(
                    timeout=timeout * 1000
                ) as download_info:

                    goto(
                        url
                    )

                return download_info.value

            except TypeError:
                with expect_download() as download_info:

                    goto(
                        url
                    )

                return download_info.value

        # Browser navigation fallback.
        if callable(goto):

            return goto(
                url
            )

        raise RuntimeError(
            "Configured browser backend does not support downloads."
        )

    # ========================================================
    # SAVE
    # ========================================================

    def _save_result(
        self,
        result: Any,
        destination: Path,
    ) -> None:

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Playwright Download object.
        save_as = getattr(
            result,
            "save_as",
            None,
        )

        if callable(save_as):

            save_as(
                str(destination)
            )

            return

        # Generic save() implementation.
        save = getattr(
            result,
            "save",
            None,
        )

        if callable(save):

            save(
                str(destination)
            )

            return

        # File path returned by browser backend.
        if isinstance(
            result,
            (
                str,
                os.PathLike,
            ),
        ):

            source = Path(
                result
            ).expanduser()

            if source.exists():

                if source.resolve() != destination.resolve():

                    shutil.copy2(
                        source,
                        destination,
                    )

                return

        # Raw bytes.
        if isinstance(
            result,
            bytes,
        ):

            destination.write_bytes(
                result
            )

            return

        # File-like object.
        read = getattr(
            result,
            "read",
            None,
        )

        if callable(read):

            data = read()

            if isinstance(
                data,
                str,
            ):
                data = data.encode(
                    "utf-8"
                )

            destination.write_bytes(
                data
            )

            return

        # Some browser backends return None after
        # downloading directly to disk.
        if result is None and destination.exists():
            return

        raise RuntimeError(
            "Unable to save downloaded content."
        )

    # ========================================================
    # PAGE
    # ========================================================

    def _get_page(
        self,
        page_id: str | None = None,
    ) -> Any:

        if self.browser_manager is None:
            raise RuntimeError(
                "BrowserManager is not configured."
            )

        getter = getattr(
            self.browser_manager,
            "get_page",
            None,
        )

        if not callable(getter):
            raise RuntimeError(
                "BrowserManager does not expose get_page()."
            )

        return getter(
            page_id
        )

    # ========================================================
    # TRACKING
    # ========================================================

    def get(
        self,
        download_id: str,
    ) -> Download | None:

        with self._lock:
            return self._downloads.get(
                download_id
            )

    def get_download(
        self,
        download_id: str,
    ) -> Download | None:

        return self.get(
            download_id
        )

    def list_downloads(
        self,
        *,
        status: str | None = None,
    ) -> list[Download]:

        with self._lock:

            downloads = list(
                self._downloads.values()
            )

        if status is not None:

            status = status.lower()

            downloads = [
                item
                for item in downloads
                if item.status == status
            ]

        return downloads

    def active_downloads(
        self,
    ) -> list[Download]:

        return [
            item
            for item in self.list_downloads()
            if item.status in {
                "pending",
                "downloading",
            }
        ]

    def completed_downloads(
        self,
    ) -> list[Download]:

        return self.list_downloads(
            status="completed"
        )

    def failed_downloads(
        self,
    ) -> list[Download]:

        return self.list_downloads(
            status="failed"
        )

    # ========================================================
    # WAIT
    # ========================================================

    def wait_for_completion(
        self,
        download_id: str,
        *,
        timeout: float = 120.0,
        poll_interval: float = 0.1,
    ) -> Download:

        deadline = (
            time.monotonic()
            + timeout
        )

        while time.monotonic() < deadline:

            record = self.get(
                download_id
            )

            if record is None:
                raise KeyError(
                    f"Download not found: {download_id}"
                )

            if record.status in {
                "completed",
                "failed",
                "cancelled",
            }:

                return record

            time.sleep(
                poll_interval
            )

        raise TimeoutError(
            f"Download did not complete within "
            f"{timeout} seconds."
        )

    # ========================================================
    # CANCEL
    # ========================================================

    def cancel(
        self,
        download_id: str,
    ) -> bool:

        record = self.get(
            download_id
        )

        if record is None:
            return False

        if record.status in {
            "completed",
            "failed",
            "cancelled",
        }:
            return False

        record.status = "cancelled"
        record.completed_at = time.time()

        return True

    # ========================================================
    # DELETE DOWNLOADED FILE
    # ========================================================

    def delete(
        self,
        download_id: str,
    ) -> bool:

        record = self.get(
            download_id
        )

        if record is None:
            return False

        path = Path(
            record.path
        )

        try:

            if path.exists():
                path.unlink()

            with self._lock:
                del self._downloads[
                    download_id
                ]

            return True

        except OSError as exc:

            logger.error(
                "Unable to delete download: %s",
                exc,
            )

            return False

    # ========================================================
    # CLEAR HISTORY
    # ========================================================

    def clear_history(
        self,
        *,
        delete_files: bool = False,
    ) -> None:

        downloads = self.list_downloads()

        if delete_files:

            for record in downloads:

                try:

                    path = Path(
                        record.path
                    )

                    if path.exists():
                        path.unlink()

                except OSError:
                    logger.warning(
                        "Could not delete %s",
                        record.path,
                    )

        with self._lock:
            self._downloads.clear()

    # ========================================================
    # FILE INFORMATION
    # ========================================================

    @staticmethod
    def _update_size(
        record: Download,
    ) -> None:

        try:

            path = Path(
                record.path
            )

            if path.exists():
                record.size = path.stat().st_size

        except OSError:

            record.size = 0

    # ========================================================
    # FILENAME
    # ========================================================

    @staticmethod
    def _filename_from_url(
        url: str,
    ) -> str:

        parsed = urlparse(
            url
        )

        path = unquote(
            parsed.path
        )

        filename = Path(
            path
        ).name

        if filename:
            return filename

        return "download"

    @staticmethod
    def _sanitize_filename(
        filename: str,
    ) -> str:

        filename = str(
            filename
        ).strip()

        if not filename:
            return "download"

        # Prevent path traversal.
        filename = filename.replace(
            "..",
            "_",
        )

        filename = filename.replace(
            "/",
            "_",
        )

        filename = filename.replace(
            "\\",
            "_",
        )

        # Windows-invalid filename characters.
        for character in '<>:"|?*':

            filename = filename.replace(
                character,
                "_",
            )

        # Avoid reserved Windows names.
        reserved = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        }

        stem = Path(
            filename
        ).stem.upper()

        if stem in reserved:
            filename = (
                "_"
                + filename
            )

        return filename

    # ========================================================
    # UNIQUE PATH
    # ========================================================

    @staticmethod
    def _unique_path(
        path: Path,
    ) -> Path:

        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix

        counter = 1

        while True:

            candidate = path.with_name(
                f"{stem}_{counter}{suffix}"
            )

            if not candidate.exists():
                return candidate

            counter += 1

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def export_history(
        self,
    ) -> list[dict[str, Any]]:

        return [
            download.to_dict()
            for download
            in self.list_downloads()
        ]


__all__ = [
    "Download",
    "DownloadManager",
]


