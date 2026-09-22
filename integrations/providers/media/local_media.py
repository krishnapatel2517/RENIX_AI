"""
RENIX Local Media Provider
==========================

Provider for media stored locally on the computer.

This provider handles discovery and metadata for local audio/video
files and delegates actual playback to an injected player adapter.

Supported concepts:

    - local media search
    - play / pause / stop
    - next / previous
    - volume
    - queue
    - seeking
    - shuffle / repeat
    - playlists
    - playback state
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

from .media_provider import MediaProvider


class LocalMediaProvider(MediaProvider):
    """Local computer media provider."""

    name = "local"

    DEFAULT_EXTENSIONS = {
        ".mp3",
        ".wav",
        ".flac",
        ".aac",
        ".m4a",
        ".ogg",
        ".opus",
        ".wma",
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".webm",
        ".wmv",
    }

    def __init__(
        self,
        *,
        media_directories: Iterable[str | Path] | None = None,
        player: Any = None,
        extensions: Iterable[str] | None = None,
    ) -> None:
        self.player = player

        self.media_directories = [
            Path(directory).expanduser()
            for directory in (
                media_directories or []
            )
        ]

        self.extensions = {
            extension.lower()
            if extension.startswith(".")
            else f".{extension.lower()}"
            for extension in (
                extensions
                or self.DEFAULT_EXTENSIONS
            )
        }

        self._queue: list[Path] = []
        self._queue_index = -1

        self._shuffle = False
        self._repeat = "off"

        self._playlists: dict[
            str,
            list[Path],
        ] = {}

    # ========================================================
    # INTERNAL
    # ========================================================

    def _require_player(self) -> Any:
        if self.player is None:
            raise RuntimeError(
                "Local media player is not configured."
            )

        return self.player

    def _call_player(
        self,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        player = self._require_player()

        function = getattr(
            player,
            method,
            None,
        )

        if not callable(function):
            raise RuntimeError(
                f"Local media player does not implement "
                f"'{method}'."
            )

        return function(
            *args,
            **kwargs,
        )

    def _is_media_file(
        self,
        path: Path,
    ) -> bool:
        return (
            path.is_file()
            and path.suffix.lower()
            in self.extensions
        )

    @staticmethod
    def _normalize_media(
        path: Path,
    ) -> dict[str, Any]:
        """Create normalized local media metadata."""

        media_type = "video"

        if path.suffix.lower() in {
            ".mp3",
            ".wav",
            ".flac",
            ".aac",
            ".m4a",
            ".ogg",
            ".opus",
            ".wma",
        }:
            media_type = "audio"

        try:
            size = path.stat().st_size
        except OSError:
            size = None

        return {
            "id": str(path.resolve()),
            "type": media_type,
            "title": path.stem,
            "filename": path.name,
            "path": str(path.resolve()),
            "extension": path.suffix.lower(),
            "size": size,
            "url": path.as_uri(),
        }

    def _resolve_media_id(
        self,
        media_id: str | Path,
    ) -> Path:
        path = Path(
            media_id
        ).expanduser()

        if not path.exists():
            raise FileNotFoundError(
                f"Media file not found: {path}"
            )

        if not self._is_media_file(
            path
        ):
            raise ValueError(
                f"Unsupported media file: {path}"
            )

        return path.resolve()

    # ========================================================
    # DISCOVERY
    # ========================================================

    def add_media_directory(
        self,
        directory: str | Path,
    ) -> None:
        """Add a directory to the media library."""

        path = Path(
            directory
        ).expanduser()

        if not path.exists():
            raise FileNotFoundError(
                f"Media directory does not exist: {path}"
            )

        if not path.is_dir():
            raise NotADirectoryError(
                str(path)
            )

        resolved = path.resolve()

        if resolved not in self.media_directories:
            self.media_directories.append(
                resolved
            )

    def remove_media_directory(
        self,
        directory: str | Path,
    ) -> bool:
        """Remove a media directory."""

        path = Path(
            directory
        ).expanduser().resolve()

        if path not in self.media_directories:
            return False

        self.media_directories.remove(
            path
        )

        return True

    def scan(
        self,
        *,
        recursive: bool = True,
    ) -> list[dict[str, Any]]:
        """Scan configured directories for media."""

        files: list[Path] = []

        for directory in self.media_directories:

            if not directory.exists():
                continue

            if recursive:
                iterator = directory.rglob("*")
            else:
                iterator = directory.glob("*")

            for path in iterator:
                if self._is_media_file(
                    path
                ):
                    files.append(
                        path.resolve()
                    )

        files.sort(
            key=lambda item: item.name.lower()
        )

        return [
            self._normalize_media(
                path
            )
            for path in files
        ]

    # ========================================================
    # PLAYBACK
    # ========================================================

    def play(
        self,
        *,
        media_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Play a local media file."""

        if media_id is not None:

            path = self._resolve_media_id(
                media_id
            )

            self._set_current_path(
                path
            )

            return self._call_player(
                "play",
                str(path),
                **kwargs,
            )

        return self._call_player(
            "resume",
            **kwargs,
        )

    def pause(
        self,
        **kwargs: Any,
    ) -> Any:
        """Pause playback."""

        return self._call_player(
            "pause",
            **kwargs,
        )

    def stop(
        self,
        **kwargs: Any,
    ) -> Any:
        """Stop playback."""

        return self._call_player(
            "stop",
            **kwargs,
        )

    def next(
        self,
        **kwargs: Any,
    ) -> Any:
        """Play next item from queue."""

        if not self._queue:
            raise RuntimeError(
                "Media queue is empty."
            )

        if self._shuffle:
            self._queue_index = random.randrange(
                len(self._queue)
            )

        else:
            self._queue_index += 1

        if self._queue_index >= len(
            self._queue
        ):
            if self._repeat == "all":
                self._queue_index = 0
            else:
                self._queue_index = (
                    len(self._queue) - 1
                )

        return self.play(
            media_id=str(
                self._queue[
                    self._queue_index
                ]
            ),
            **kwargs,
        )

    def previous(
        self,
        **kwargs: Any,
    ) -> Any:
        """Play previous item from queue."""

        if not self._queue:
            raise RuntimeError(
                "Media queue is empty."
            )

        self._queue_index -= 1

        if self._queue_index < 0:
            if self._repeat == "all":
                self._queue_index = (
                    len(self._queue) - 1
                )
            else:
                self._queue_index = 0

        return self.play(
            media_id=str(
                self._queue[
                    self._queue_index
                ]
            ),
            **kwargs,
        )

    # ========================================================
    # NOW PLAYING
    # ========================================================

    def now_playing(
        self,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Return current local media state."""

        player = self._require_player()

        getter = getattr(
            player,
            "now_playing",
            None,
        )

        if callable(getter):
            result = getter(
                **kwargs
            )

            if not result:
                return None

            if isinstance(
                result,
                dict,
            ):
                return result

        current = getattr(
            player,
            "current",
            None,
        )

        if current:
            try:
                return self._normalize_media(
                    Path(current)
                )
            except Exception:
                return {
                    "id": str(current),
                    "title": Path(
                        str(current)
                    ).stem,
                    "path": str(current),
                }

        if (
            self._queue
            and 0 <= self._queue_index
            < len(self._queue)
        ):
            return self._normalize_media(
                self._queue[
                    self._queue_index
                ]
            )

        return None

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        *,
        media_type: str | None = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search local media."""

        query = query.strip().lower()

        if not query or limit <= 0:
            return []

        results = []

        for item in self.scan(
            recursive=True
        ):
            title = str(
                item.get(
                    "title",
                    "",
                )
            ).lower()

            filename = str(
                item.get(
                    "filename",
                    "",
                )
            ).lower()

            if (
                query not in title
                and query not in filename
            ):
                continue

            if (
                media_type
                and item.get("type")
                != media_type
            ):
                continue

            results.append(
                item
            )

            if len(results) >= limit:
                break

        return results

    # ========================================================
    # VOLUME
    # ========================================================

    def get_volume(
        self,
        **kwargs: Any,
    ) -> float:
        """Return player volume."""

        result = self._call_player(
            "get_volume",
            **kwargs,
        )

        value = float(
            result
        )

        if value > 1.0:
            value /= 100.0

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    def set_volume(
        self,
        volume: float,
        **kwargs: Any,
    ) -> Any:
        """Set player volume."""

        if not 0.0 <= volume <= 1.0:
            raise ValueError(
                "volume must be between 0.0 and 1.0."
            )

        return self._call_player(
            "set_volume",
            volume,
            **kwargs,
        )

    # ========================================================
    # QUEUE
    # ========================================================

    def get_queue(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return current queue."""

        return [
            self._normalize_media(
                path
            )
            for path in self._queue
        ]

    def add_to_queue(
        self,
        media_id: str,
        **kwargs: Any,
    ) -> Any:
        """Add a local media file to the queue."""

        path = self._resolve_media_id(
            media_id
        )

        self._queue.append(
            path
        )

        if self._queue_index == -1:
            self._queue_index = 0

        return self._normalize_media(
            path
        )

    def clear_queue(
        self,
        **kwargs: Any,
    ) -> Any:
        """Clear the playback queue."""

        self._queue.clear()
        self._queue_index = -1

        return True

    def _set_current_path(
        self,
        path: Path,
    ) -> None:
        """Set current path in the local queue."""

        try:
            self._queue_index = (
                self._queue.index(path)
            )
        except ValueError:
            self._queue.append(
                path
            )
            self._queue_index = (
                len(self._queue) - 1
            )

    # ========================================================
    # SEEK
    # ========================================================

    def get_position(
        self,
        **kwargs: Any,
    ) -> float:
        """Return current playback position."""

        return float(
            self._call_player(
                "get_position",
                **kwargs,
            )
        )

    def seek(
        self,
        position: float,
        **kwargs: Any,
    ) -> Any:
        """Seek playback."""

        if position < 0:
            raise ValueError(
                "position cannot be negative."
            )

        return self._call_player(
            "seek",
            position,
            **kwargs,
        )

    # ========================================================
    # SHUFFLE / REPEAT
    # ========================================================

    def set_shuffle(
        self,
        enabled: bool,
        **kwargs: Any,
    ) -> Any:
        """Enable or disable queue shuffle."""

        self._shuffle = bool(
            enabled
        )

        player = self.player

        if player is not None:
            method = getattr(
                player,
                "set_shuffle",
                None,
            )

            if callable(method):
                return method(
                    self._shuffle,
                    **kwargs,
                )

        return self._shuffle

    def set_repeat(
        self,
        mode: str,
        **kwargs: Any,
    ) -> Any:
        """Set repeat mode."""

        mode = mode.lower().strip()

        if mode not in {
            "off",
            "one",
            "all",
        }:
            raise ValueError(
                "repeat mode must be off, one, or all."
            )

        self._repeat = mode

        player = self.player

        if player is not None:
            method = getattr(
                player,
                "set_repeat",
                None,
            )

            if callable(method):
                return method(
                    mode,
                    **kwargs,
                )

        return mode

    # ========================================================
    # PLAYLISTS
    # ========================================================

    def list_playlists(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """List local playlists."""

        return [
            {
                "id": name,
                "name": name,
                "type": "local",
                "tracks_total": len(
                    tracks
                ),
            }
            for name, tracks
            in self._playlists.items()
        ]

    def get_playlist(
        self,
        playlist_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a local playlist."""

        if playlist_id not in self._playlists:
            raise KeyError(
                f"Playlist not found: "
                f"{playlist_id}"
            )

        tracks = self._playlists[
            playlist_id
        ]

        return {
            "id": playlist_id,
            "name": playlist_id,
            "type": "local",
            "tracks": [
                self._normalize_media(
                    path
                )
                for path in tracks
            ],
            "tracks_total": len(
                tracks
            ),
        }

    def create_playlist(
        self,
        name: str,
        *,
        description: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a local playlist."""

        name = name.strip()

        if not name:
            raise ValueError(
                "Playlist name cannot be empty."
            )

        if name in self._playlists:
            raise ValueError(
                f"Playlist already exists: {name}"
            )

        self._playlists[name] = []

        return {
            "id": name,
            "name": name,
            "description": description,
            "type": "local",
            "tracks": [],
            "tracks_total": 0,
        }

    def add_to_playlist(
        self,
        playlist_id: str,
        media_ids: list[str],
        **kwargs: Any,
    ) -> Any:
        """Add media to a local playlist."""

        if playlist_id not in self._playlists:
            raise KeyError(
                f"Playlist not found: "
                f"{playlist_id}"
            )

        added = []

        for media_id in media_ids:
            path = self._resolve_media_id(
                media_id
            )

            if path not in self._playlists[
                playlist_id
            ]:
                self._playlists[
                    playlist_id
                ].append(
                    path
                )

                added.append(
                    self._normalize_media(
                        path
                    )
                )

        return added

    def remove_from_playlist(
        self,
        playlist_id: str,
        media_ids: list[str],
        **kwargs: Any,
    ) -> Any:
        """Remove media from a local playlist."""

        if playlist_id not in self._playlists:
            raise KeyError(
                f"Playlist not found: "
                f"{playlist_id}"
            )

        removed = []

        for media_id in media_ids:

            path = self._resolve_media_id(
                media_id
            )

            if path in self._playlists[
                playlist_id
            ]:
                self._playlists[
                    playlist_id
                ].remove(
                    path
                )

                removed.append(
                    self._normalize_media(
                        path
                    )
                )

        return removed

    # ========================================================
    # DEVICES
    # ========================================================

    def list_devices(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return local playback devices."""

        player = self.player

        if player is None:
            return []

        method = getattr(
            player,
            "list_devices",
            None,
        )

        if callable(method):
            return list(
                method(
                    **kwargs
                )
                or []
            )

        return []

    def set_device(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Select a local playback device."""

        return self._call_player(
            "set_device",
            device_id,
            **kwargs,
        )

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(
        self,
    ) -> bool:
        """Check local media provider."""

        try:
            if not self.media_directories:
                return self.player is not None

            return any(
                directory.exists()
                and directory.is_dir()
                for directory
                in self.media_directories
            )

        except Exception:
            return False

    # ========================================================
    # CAPABILITIES
    # ========================================================

    def capabilities(
        self,
    ) -> dict[str, bool]:
        """Return local media capabilities."""

        return {
            "playback": True,
            "pause": True,
            "stop": True,
            "next": True,
            "previous": True,
            "volume": True,
            "search": True,
            "playlists": True,
            "playlist_editing": True,
            "queue": True,
            "queue_clear": True,
            "seek": True,
            "shuffle": True,
            "repeat": True,
            "devices": True,
        }


__all__ = [
    "LocalMediaProvider",
]


