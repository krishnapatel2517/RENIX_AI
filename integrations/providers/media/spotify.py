"""
RENIX Spotify Media Provider
============================

Spotify integration for RENIX.

This provider uses an injected Spotify-compatible client. Authentication,
OAuth token management, and API client construction remain outside this
module.

The provider translates Spotify operations into the common RENIX
MediaProvider interface.
"""

from __future__ import annotations

from typing import Any

from .media_provider import MediaProvider


class SpotifyProvider(MediaProvider):
    """Spotify media provider."""

    name = "spotify"

    def __init__(
        self,
        *,
        client: Any = None,
    ) -> None:
        self.client = client

    # ========================================================
    # INTERNAL
    # ========================================================

    def _require_client(self) -> Any:
        """Return the configured Spotify client."""

        if self.client is None:
            raise RuntimeError(
                "Spotify client is not configured."
            )

        return self.client

    def _call(
        self,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call a method on the Spotify client."""

        client = self._require_client()

        function = getattr(
            client,
            method,
            None,
        )

        if not callable(function):
            raise RuntimeError(
                f"Spotify client does not implement "
                f"'{method}'."
            )

        return function(
            *args,
            **kwargs,
        )

    @staticmethod
    def _normalize_track(
        track: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize Spotify track information."""

        artists = track.get(
            "artists",
            [],
        )

        return {
            "id": track.get("id"),
            "type": "track",
            "title": track.get(
                "name",
                "",
            ),
            "artist": ", ".join(
                artist.get(
                    "name",
                    "",
                )
                for artist in artists
                if isinstance(
                    artist,
                    dict,
                )
            ),
            "album": (
                track.get(
                    "album",
                    {}
                ).get(
                    "name"
                )
                if isinstance(
                    track.get(
                        "album",
                        {},
                    ),
                    dict,
                )
                else None
            ),
            "duration_ms": track.get(
                "duration_ms"
            ),
            "uri": track.get(
                "uri"
            ),
            "url": (
                track.get(
                    "external_urls",
                    {},
                ).get(
                    "spotify"
                )
                if isinstance(
                    track.get(
                        "external_urls",
                        {},
                    ),
                    dict,
                )
                else None
            ),
            "raw": track,
        }

    @staticmethod
    def _normalize_playlist(
        playlist: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize Spotify playlist."""

        owner = playlist.get(
            "owner",
            {},
        )

        return {
            "id": playlist.get(
                "id"
            ),
            "name": playlist.get(
                "name",
                "",
            ),
            "description": playlist.get(
                "description",
                "",
            ),
            "public": playlist.get(
                "public"
            ),
            "owner": (
                owner.get(
                    "display_name"
                )
                if isinstance(
                    owner,
                    dict,
                )
                else None
            ),
            "tracks_total": (
                playlist.get(
                    "tracks",
                    {},
                ).get(
                    "total"
                )
                if isinstance(
                    playlist.get(
                        "tracks",
                        {},
                    ),
                    dict,
                )
                else None
            ),
            "url": (
                playlist.get(
                    "external_urls",
                    {},
                ).get(
                    "spotify"
                )
                if isinstance(
                    playlist.get(
                        "external_urls",
                        {},
                    ),
                    dict,
                )
                else None
            ),
            "raw": playlist,
        }

    @classmethod
    def _normalize_item(
        cls,
        item: Any,
    ) -> dict[str, Any]:
        """Normalize a Spotify media item."""

        if not isinstance(
            item,
            dict,
        ):
            return {
                "id": None,
                "type": "unknown",
                "title": str(item),
                "raw": item,
            }

        item_type = item.get(
            "type"
        )

        if item_type == "track":
            return cls._normalize_track(
                item
            )

        if item_type == "playlist":
            return cls._normalize_playlist(
                item
            )

        return {
            "id": item.get(
                "id"
            ),
            "type": item_type,
            "title": item.get(
                "name",
                "",
            ),
            "uri": item.get(
                "uri"
            ),
            "raw": item,
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
        """Start or resume Spotify playback."""

        if media_id is not None:

            uri = media_id

            if not uri.startswith(
                "spotify:"
            ):
                uri = (
                    "spotify:track:"
                    + media_id
                )

            return self._call(
                "start_playback",
                uris=[uri],
                **kwargs,
            )

        return self._call(
            "start_playback",
            **kwargs,
        )

    def pause(
        self,
        **kwargs: Any,
    ) -> Any:
        """Pause Spotify playback."""

        return self._call(
            "pause_playback",
            **kwargs,
        )

    def stop(
        self,
        **kwargs: Any,
    ) -> Any:
        """
        Stop playback.

        Spotify's Web API does not expose a dedicated stop endpoint,
        so RENIX pauses playback.
        """

        return self._call(
            "pause_playback",
            **kwargs,
        )

    def next(
        self,
        **kwargs: Any,
    ) -> Any:
        """Skip to next Spotify track."""

        return self._call(
            "next_track",
            **kwargs,
        )

    def previous(
        self,
        **kwargs: Any,
    ) -> Any:
        """Skip to previous Spotify track."""

        return self._call(
            "previous_track",
            **kwargs,
        )

    # ========================================================
    # VOLUME
    # ========================================================

    def get_volume(
        self,
        **kwargs: Any,
    ) -> float:
        """Get current Spotify device volume."""

        playback = self._call(
            "current_playback",
            **kwargs,
        )

        if not playback:
            return 0.0

        device = playback.get(
            "device",
            {},
        )

        volume = device.get(
            "volume_percent",
            0,
        )

        return max(
            0.0,
            min(
                1.0,
                float(volume) / 100.0,
            ),
        )

    def set_volume(
        self,
        volume: float,
        **kwargs: Any,
    ) -> Any:
        """Set Spotify volume."""

        if not 0.0 <= volume <= 1.0:
            raise ValueError(
                "volume must be between 0.0 and 1.0."
            )

        return self._call(
            "volume",
            int(
                round(
                    volume * 100
                )
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
        """Return the currently playing Spotify item."""

        playback = self._call(
            "current_playback",
            **kwargs,
        )

        if not playback:
            return None

        item = playback.get(
            "item"
        )

        if not item:
            return None

        result = self._normalize_item(
            item
        )

        result.update(
            {
                "is_playing": playback.get(
                    "is_playing",
                    False,
                ),
                "progress_ms": playback.get(
                    "progress_ms",
                    0,
                ),
                "device": playback.get(
                    "device"
                ),
                "shuffle": playback.get(
                    "shuffle_state"
                ),
                "repeat": playback.get(
                    "repeat_state"
                ),
            }
        )

        return result

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
        """Search Spotify."""

        if not query.strip():
            return []

        if limit <= 0:
            return []

        types = (
            media_type
            if media_type
            else "track,album,artist,playlist"
        )

        response = self._call(
            "search",
            q=query,
            type=types,
            limit=min(
                limit,
                50,
            ),
            **kwargs,
        )

        results: list[dict[str, Any]] = []

        for key in (
            "tracks",
            "albums",
            "artists",
            "playlists",
        ):
            collection = response.get(
                key,
                {},
            )

            if not isinstance(
                collection,
                dict,
            ):
                continue

            for item in collection.get(
                "items",
                [],
            ):
                results.append(
                    self._normalize_item(
                        item
                    )
                )

        return results[:limit]

    # ========================================================
    # PLAYLISTS
    # ========================================================

    def list_playlists(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return Spotify playlists."""

        response = self._call(
            "current_user_playlists",
            **kwargs,
        )

        return [
            self._normalize_playlist(
                playlist
            )
            for playlist in response.get(
                "items",
                [],
            )
        ]

    def get_playlist(
        self,
        playlist_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a Spotify playlist."""

        playlist = self._call(
            "playlist",
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
        """Create a Spotify playlist."""

        if not name.strip():
            raise ValueError(
                "Playlist name cannot be empty."
            )

        user_id = kwargs.pop(
            "user_id",
            None,
        )

        if user_id is None:

            profile = self._call(
                "current_user"
            )

            user_id = profile.get(
                "id"
            )

        playlist = self._call(
            "user_playlist_create",
            user_id,
            name,
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
        """Add tracks to a Spotify playlist."""

        uris = []

        for media_id in media_ids:

            if media_id.startswith(
                "spotify:"
            ):
                uris.append(
                    media_id
                )
            else:
                uris.append(
                    "spotify:track:"
                    + media_id
                )

        return self._call(
            "playlist_add_items",
            playlist_id,
            uris,
            **kwargs,
        )

    def remove_from_playlist(
        self,
        playlist_id: str,
        media_ids: list[str],
        **kwargs: Any,
    ) -> Any:
        """Remove tracks from a Spotify playlist."""

        tracks = []

        for media_id in media_ids:

            uri = media_id

            if not uri.startswith(
                "spotify:"
            ):
                uri = (
                    "spotify:track:"
                    + uri
                )

            tracks.append(
                {
                    "uri": uri
                }
            )

        return self._call(
            "playlist_remove_all_occurrences_of_items",
            playlist_id,
            tracks,
            **kwargs,
        )

    # ========================================================
    # QUEUE
    # ========================================================

    def get_queue(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return Spotify playback queue."""

        response = self._call(
            "queue",
            **kwargs,
        )

        return [
            self._normalize_item(
                item
            )
            for item in response.get(
                "queue",
                [],
            )
        ]

    def add_to_queue(
        self,
        media_id: str,
        **kwargs: Any,
    ) -> Any:
        """Add a track to Spotify queue."""

        uri = media_id

        if not uri.startswith(
            "spotify:"
        ):
            uri = (
                "spotify:track:"
                + uri
            )

        return self._call(
            "add_to_queue",
            uri,
            **kwargs,
        )

    def clear_queue(
        self,
        **kwargs: Any,
    ) -> Any:
        """
        Spotify does not provide a direct queue-clear operation.

        This method intentionally raises rather than pretending the
        queue was cleared.
        """

        raise NotImplementedError(
            "Spotify does not provide a direct queue-clear API."
        )

    # ========================================================
    # SEEK
    # ========================================================

    def get_position(
        self,
        **kwargs: Any,
    ) -> float:
        """Return current playback position in seconds."""

        playback = self._call(
            "current_playback",
            **kwargs,
        )

        if not playback:
            return 0.0

        return float(
            playback.get(
                "progress_ms",
                0,
            )
        ) / 1000.0

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

        return self._call(
            "seek_track",
            int(
                position * 1000
            ),
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
        """Enable or disable Spotify shuffle."""

        return self._call(
            "shuffle",
            enabled,
            **kwargs,
        )

    def set_repeat(
        self,
        mode: str,
        **kwargs: Any,
    ) -> Any:
        """Set Spotify repeat mode."""

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

        return self._call(
            "repeat",
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
        """Return available Spotify devices."""

        response = self._call(
            "devices",
            **kwargs,
        )

        return response.get(
            "devices",
            [],
        )

    def set_device(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Transfer playback to a Spotify device."""

        return self._call(
            "transfer_playback",
            [device_id],
            **kwargs,
        )

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(
        self,
    ) -> bool:
        """Check Spotify connectivity."""

        try:
            self._call(
                "current_user"
            )
            return True

        except Exception:
            return False

    # ========================================================
    # CAPABILITIES
    # ========================================================

    def capabilities(
        self,
    ) -> dict[str, bool]:
        """Return Spotify capabilities."""

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
            "queue_clear": False,
            "seek": True,
            "shuffle": True,
            "repeat": True,
            "devices": True,
        }


__all__ = [
    "SpotifyProvider",
]


