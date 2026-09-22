"""
RENIX YouTube Media Provider
============================

YouTube integration for RENIX.

This provider uses an injected YouTube-compatible client/adapter.
Authentication, API credentials, browser automation, and playback
implementation remain outside this module.

The provider exposes a common RENIX MediaProvider interface.
"""

from __future__ import annotations

from typing import Any

from .media_provider import MediaProvider


class YouTubeProvider(MediaProvider):
    """YouTube media provider."""

    name = "youtube"

    def __init__(
        self,
        *,
        client: Any = None,
        player: Any = None,
    ) -> None:
        self.client = client
        self.player = player

    # ========================================================
    # INTERNAL
    # ========================================================

    def _require_client(self) -> Any:
        if self.client is None:
            raise RuntimeError(
                "YouTube client is not configured."
            )

        return self.client

    def _require_player(self) -> Any:
        if self.player is None:
            raise RuntimeError(
                "YouTube player is not configured."
            )

        return self.player

    def _call_client(
        self,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        client = self._require_client()

        function = getattr(
            client,
            method,
            None,
        )

        if not callable(function):
            raise RuntimeError(
                f"YouTube client does not implement "
                f"'{method}'."
            )

        return function(
            *args,
            **kwargs,
        )

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
                f"YouTube player does not implement "
                f"'{method}'."
            )

        return function(
            *args,
            **kwargs,
        )

    @staticmethod
    def _normalize_video(
        video: Any,
    ) -> dict[str, Any]:
        """Normalize YouTube video information."""

        if not isinstance(
            video,
            dict,
        ):
            return {
                "id": None,
                "type": "video",
                "title": str(video),
                "raw": video,
            }

        video_id = video.get(
            "id"
        )

        if isinstance(
            video_id,
            dict,
        ):
            video_id = video_id.get(
                "videoId"
            )

        snippet = video.get(
            "snippet",
            {},
        )

        if not isinstance(
            snippet,
            dict,
        ):
            snippet = {}

        return {
            "id": video_id,
            "type": "video",
            "title": snippet.get(
                "title",
                video.get(
                    "title",
                    "",
                ),
            ),
            "description": snippet.get(
                "description",
                video.get(
                    "description",
                    "",
                ),
            ),
            "channel": snippet.get(
                "channelTitle"
            ),
            "published_at": snippet.get(
                "publishedAt"
            ),
            "thumbnail": (
                video.get(
                    "thumbnail"
                )
                or snippet.get(
                    "thumbnails",
                    {},
                )
            ),
            "url": (
                f"https://www.youtube.com/watch?v={video_id}"
                if video_id
                else video.get("url")
            ),
            "duration": video.get(
                "duration"
            ),
            "raw": video,
        }

    @staticmethod
    def _normalize_playlist(
        playlist: Any,
    ) -> dict[str, Any]:
        """Normalize YouTube playlist information."""

        if not isinstance(
            playlist,
            dict,
        ):
            return {
                "id": None,
                "type": "playlist",
                "name": str(playlist),
                "raw": playlist,
            }

        playlist_id = playlist.get(
            "id"
        )

        if isinstance(
            playlist_id,
            dict,
        ):
            playlist_id = playlist_id.get(
                "playlistId"
            )

        snippet = playlist.get(
            "snippet",
            {},
        )

        if not isinstance(
            snippet,
            dict,
        ):
            snippet = {}

        return {
            "id": playlist_id,
            "type": "playlist",
            "name": snippet.get(
                "title",
                playlist.get(
                    "title",
                    "",
                ),
            ),
            "description": snippet.get(
                "description",
                playlist.get(
                    "description",
                    "",
                ),
            ),
            "channel": snippet.get(
                "channelTitle"
            ),
            "url": (
                f"https://www.youtube.com/playlist?list={playlist_id}"
                if playlist_id
                else playlist.get("url")
            ),
            "raw": playlist,
        }

    # ========================================================
    # PLAYBACK
    # ========================================================

    def play(
        self,
        *,
        media_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Play a YouTube video."""

        player = self._require_player()

        if media_id is not None:

            video_id = media_id

            if "v=" in video_id:
                video_id = (
                    video_id.split(
                        "v=",
                        1,
                    )[1]
                    .split(
                        "&",
                        1,
                    )[0]
                )

            if video_id.startswith(
                "http://"
            ) or video_id.startswith(
                "https://"
            ):
                url = video_id
            else:
                url = (
                    "https://www.youtube.com/watch?v="
                    + video_id
                )

            load = getattr(
                player,
                "load",
                None,
            )

            if callable(load):
                load(
                    url,
                    **kwargs,
                )

        play = getattr(
            player,
            "play",
            None,
        )

        if not callable(play):
            raise RuntimeError(
                "YouTube player does not implement 'play'."
            )

        return play(
            **kwargs
        )

    def pause(
        self,
        **kwargs: Any,
    ) -> Any:
        """Pause YouTube playback."""

        return self._call_player(
            "pause",
            **kwargs,
        )

    def stop(
        self,
        **kwargs: Any,
    ) -> Any:
        """Stop YouTube playback."""

        return self._call_player(
            "stop",
            **kwargs,
        )

    def next(
        self,
        **kwargs: Any,
    ) -> Any:
        """Play the next queued YouTube item."""

        return self._call_player(
            "next",
            **kwargs,
        )

    def previous(
        self,
        **kwargs: Any,
    ) -> Any:
        """Play the previous YouTube item."""

        return self._call_player(
            "previous",
            **kwargs,
        )

    # ========================================================
    # VOLUME
    # ========================================================

    def get_volume(
        self,
        **kwargs: Any,
    ) -> float:
        """Return volume from 0.0 to 1.0."""

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
        """Set YouTube player volume."""

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
    # NOW PLAYING
    # ========================================================

    def now_playing(
        self,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Return current YouTube playback information."""

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

        else:
            result = {}

            for key, method_name in (
                (
                    "id",
                    "get_video_id",
                ),
                (
                    "title",
                    "get_title",
                ),
                (
                    "position",
                    "get_position",
                ),
                (
                    "duration",
                    "get_duration",
                ),
                (
                    "is_playing",
                    "is_playing",
                ),
            ):
                method = getattr(
                    player,
                    method_name,
                    None,
                )

                if callable(method):
                    result[key] = method()

        if not result:
            return None

        if isinstance(
            result,
            dict,
        ):
            return {
                "id": result.get(
                    "id"
                ),
                "type": "video",
                "title": result.get(
                    "title",
                    "",
                ),
                "position": result.get(
                    "position",
                    0,
                ),
                "duration": result.get(
                    "duration"
                ),
                "is_playing": result.get(
                    "is_playing",
                    False,
                ),
                "url": (
                    f"https://www.youtube.com/watch?v={result.get('id')}"
                    if result.get("id")
                    else None
                ),
                "raw": result,
            }

        return {
            "id": None,
            "type": "video",
            "title": str(result),
            "raw": result,
        }

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
        """Search YouTube."""

        if not query.strip():
            return []

        if limit <= 0:
            return []

        response = self._call_client(
            "search",
            query=query,
            media_type=media_type or "video",
            limit=min(
                limit,
                50,
            ),
            **kwargs,
        )

        if isinstance(
            response,
            dict,
        ):
            items = response.get(
                "items",
                [],
            )
        else:
            items = response or []

        return [
            self._normalize_video(
                item
            )
            for item in items
        ][:limit]

    # ========================================================
    # PLAYLISTS
    # ========================================================

    def list_playlists(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """List YouTube playlists."""

        response = self._call_client(
            "list_playlists",
            **kwargs,
        )

        if isinstance(
            response,
            dict,
        ):
            response = response.get(
                "items",
                [],
            )

        return [
            self._normalize_playlist(
                item
            )
            for item in (
                response or []
            )
        ]

    def get_playlist(
        self,
        playlist_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get a YouTube playlist."""

        playlist = self._call_client(
            "get_playlist",
            playlist_id,
            **kwargs,
        )

        return self._normalize_playlist(
            playlist
        )

    def create_playlist(
        self,
        name: str,
        *,
        description: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a YouTube playlist."""

        if not name.strip():
            raise ValueError(
                "Playlist name cannot be empty."
            )

        playlist = self._call_client(
            "create_playlist",
            name=name,
            description=description,
            **kwargs,
        )

        return self._normalize_playlist(
            playlist
        )

    def add_to_playlist(
        self,
        playlist_id: str,
        media_ids: list[str],
        **kwargs: Any,
    ) -> Any:
        """Add videos to a YouTube playlist."""

        if not media_ids:
            return None

        return self._call_client(
            "add_to_playlist",
            playlist_id=playlist_id,
            video_ids=media_ids,
            **kwargs,
        )

    def remove_from_playlist(
        self,
        playlist_id: str,
        media_ids: list[str],
        **kwargs: Any,
    ) -> Any:
        """Remove videos from a YouTube playlist."""

        if not media_ids:
            return None

        return self._call_client(
            "remove_from_playlist",
            playlist_id=playlist_id,
            video_ids=media_ids,
            **kwargs,
        )

    # ========================================================
    # QUEUE
    # ========================================================

    def get_queue(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return current YouTube queue."""

        result = self._call_player(
            "get_queue",
            **kwargs,
        )

        return [
            self._normalize_video(
                item
            )
            for item in (
                result or []
            )
        ]

    def add_to_queue(
        self,
        media_id: str,
        **kwargs: Any,
    ) -> Any:
        """Add a video to the YouTube queue."""

        return self._call_player(
            "add_to_queue",
            media_id,
            **kwargs,
        )

    def clear_queue(
        self,
        **kwargs: Any,
    ) -> Any:
        """Clear YouTube queue."""

        return self._call_player(
            "clear_queue",
            **kwargs,
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
        """Seek to a playback position."""

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
        """Enable or disable shuffle."""

        return self._call_player(
            "set_shuffle",
            enabled,
            **kwargs,
        )

    def set_repeat(
        self,
        mode: str,
        **kwargs: Any,
    ) -> Any:
        """Set repeat mode."""

        allowed = {
            "off",
            "one",
            "all",
        }

        mode = mode.lower().strip()

        if mode not in allowed:
            raise ValueError(
                "repeat mode must be off, one, or all."
            )

        return self._call_player(
            "set_repeat",
            mode,
            **kwargs,
        )

    # ========================================================
    # DEVICES
    # ========================================================

    def list_devices(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return available YouTube playback devices."""

        return self._call_player(
            "list_devices",
            **kwargs,
        )

    def set_device(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Select a YouTube playback device."""

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
        """Check YouTube provider health."""

        try:
            if self.client is not None:
                health = getattr(
                    self.client,
                    "health_check",
                    None,
                )

                if callable(health):
                    return bool(
                        health()
                    )

            if self.player is not None:
                return True

            return False

        except Exception:
            return False

    # ========================================================
    # CAPABILITIES
    # ========================================================

    def capabilities(
        self,
    ) -> dict[str, bool]:
        """Return YouTube capabilities."""

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
    "YouTubeProvider",
]


