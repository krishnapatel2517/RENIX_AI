"""
RENIX Music Manager

Handles music discovery, playback requests, search,
favorites, recently played tracks, queue management,
and music metadata.

Actual playback can be connected later to Spotify,
YouTube Music, local files, VLC, or another provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .media_manager import MediaManager


@dataclass
class Track:
    """Represents a music track."""

    title: str
    artist: str = "Unknown Artist"
    album: str = ""
    source: str = ""
    duration: float = 0.0
    genre: str = ""
    year: int | None = None
    track_id: str | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.source:
            self.source = self.title

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "source": self.source,
            "duration": self.duration,
            "genre": self.genre,
            "year": self.year,
            "track_id": self.track_id,
            "metadata": dict(self.metadata),
        }


class MusicManager:
    """
    Music subsystem for RENIX.

    This class intentionally separates music logic from
    the actual playback backend.
    """

    def __init__(
        self,
        media_manager: MediaManager | None = None,
    ) -> None:

        self.media_manager = (
            media_manager
            or MediaManager()
        )

        self.library: dict[str, Track] = {}

        self.favorites: set[str] = set()

        self.recently_played: list[str] = []

        self.search_history: list[str] = []

        self._track_counter = 0

    # ============================================================
    # LIBRARY
    # ============================================================

    def add_track(
        self,
        track: Track,
    ) -> str:

        if not isinstance(track, Track):
            raise TypeError(
                "track must be a Track instance."
            )

        if not track.track_id:
            self._track_counter += 1

            track.track_id = (
                f"TRACK-{self._track_counter:06d}"
            )

        self.library[
            track.track_id
        ] = track

        return track.track_id

    def add_tracks(
        self,
        tracks: Iterable[Track],
    ) -> list[str]:

        ids = []

        for track in tracks:
            ids.append(
                self.add_track(track)
            )

        return ids

    def remove_track(
        self,
        track_id: str,
    ) -> bool:

        if track_id not in self.library:
            return False

        self.library.pop(
            track_id
        )

        self.favorites.discard(
            track_id
        )

        self.recently_played = [
            item
            for item in self.recently_played
            if item != track_id
        ]

        return True

    def get_track(
        self,
        track_id: str,
    ) -> Track | None:

        return self.library.get(
            track_id
        )

    def get_library(
        self,
    ) -> list[dict[str, Any]]:

        return [
            track.to_dict()
            for track in self.library.values()
        ]

    def clear_library(self) -> None:

        self.library.clear()
        self.favorites.clear()
        self.recently_played.clear()

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:

        query = (
            query or ""
        ).strip().casefold()

        if not query:
            return []

        self.search_history.append(
            query
        )

        if len(self.search_history) > 100:
            self.search_history = (
                self.search_history[-100:]
            )

        results = []

        for track in self.library.values():

            searchable = " ".join(
                [
                    track.title,
                    track.artist,
                    track.album,
                    track.genre,
                    str(track.year or ""),
                ]
            ).casefold()

            if query in searchable:

                results.append(
                    track.to_dict()
                )

        return results[:max(0, limit)]

    def search_artist(
        self,
        artist: str,
    ) -> list[dict[str, Any]]:

        artist = (
            artist or ""
        ).strip().casefold()

        return [
            track.to_dict()
            for track in self.library.values()
            if artist in track.artist.casefold()
        ]

    def search_album(
        self,
        album: str,
    ) -> list[dict[str, Any]]:

        album = (
            album or ""
        ).strip().casefold()

        return [
            track.to_dict()
            for track in self.library.values()
            if album in track.album.casefold()
        ]

    def search_genre(
        self,
        genre: str,
    ) -> list[dict[str, Any]]:

        genre = (
            genre or ""
        ).strip().casefold()

        return [
            track.to_dict()
            for track in self.library.values()
            if genre in track.genre.casefold()
        ]

    # ============================================================
    # PLAYBACK
    # ============================================================

    def play(
        self,
        track: Track | str,
    ) -> dict[str, Any]:

        if isinstance(track, str):

            found = self.get_track(
                track
            )

            if found is None:
                return {
                    "success": False,
                    "message": (
                        f"Track '{track}' "
                        "was not found."
                    ),
                }

            track = found

        if not isinstance(track, Track):
            return {
                "success": False,
                "message": "Invalid track.",
            }

        result = self.media_manager.play(
            track.source,
            title=track.title,
            media_type="music",
            duration=track.duration,
        )

        if track.track_id:

            self.recently_played.append(
                track.track_id
            )

            if len(
                self.recently_played
            ) > 100:

                self.recently_played = (
                    self.recently_played[-100:]
                )

        return {
            "success": True,
            "track": track.to_dict(),
            "media_status": result,
        }

    def pause(self) -> dict[str, Any]:

        return self.media_manager.pause()

    def resume(self) -> dict[str, Any]:

        return self.media_manager.resume()

    def stop(self) -> dict[str, Any]:

        return self.media_manager.stop()

    def next(self) -> dict[str, Any]:

        return self.media_manager.play_next()

    def previous(self) -> dict[str, Any]:

        return self.media_manager.play_previous()

    # ============================================================
    # QUEUE
    # ============================================================

    def add_to_queue(
        self,
        track: Track | str,
    ) -> dict[str, Any]:

        if isinstance(track, str):
            track = self.get_track(track)

        if not isinstance(track, Track):
            return {
                "success": False,
                "message": "Track not found.",
            }

        result = self.media_manager.add_to_queue(
            track.source,
            title=track.title,
            media_type="music",
            duration=track.duration,
        )

        return {
            **result,
            "track": track.to_dict(),
        }

    def queue_tracks(
        self,
        tracks: Iterable[Track | str],
    ) -> list[dict[str, Any]]:

        results = []

        for track in tracks:
            results.append(
                self.add_to_queue(track)
            )

        return results

    def get_queue(
        self,
    ) -> list[dict[str, Any]]:

        return self.media_manager.get_queue()

    def clear_queue(self) -> dict[str, Any]:

        return self.media_manager.clear_queue()

    # ============================================================
    # FAVORITES
    # ============================================================

    def add_favorite(
        self,
        track_id: str,
    ) -> bool:

        if track_id not in self.library:
            return False

        self.favorites.add(
            track_id
        )

        return True

    def remove_favorite(
        self,
        track_id: str,
    ) -> bool:

        if track_id not in self.favorites:
            return False

        self.favorites.remove(
            track_id
        )

        return True

    def toggle_favorite(
        self,
        track_id: str,
    ) -> bool:

        if track_id in self.favorites:

            self.favorites.remove(
                track_id
            )

            return False

        if track_id in self.library:

            self.favorites.add(
                track_id
            )

            return True

        return False

    def is_favorite(
        self,
        track_id: str,
    ) -> bool:

        return track_id in self.favorites

    def get_favorites(
        self,
    ) -> list[dict[str, Any]]:

        return [
            self.library[
                track_id
            ].to_dict()
            for track_id in self.favorites
            if track_id in self.library
        ]

    # ============================================================
    # RECENTLY PLAYED
    # ============================================================

    def get_recently_played(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:

        limit = max(
            0,
            int(limit),
        )

        result = []

        for track_id in reversed(
            self.recently_played
        ):

            track = self.library.get(
                track_id
            )

            if track:
                result.append(
                    track.to_dict()
                )

            if len(result) >= limit:
                break

        return result

    def clear_recently_played(
        self,
    ) -> None:

        self.recently_played.clear()

    # ============================================================
    # RECOMMENDATIONS
    # ============================================================

    def recommend(
        self,
        *,
        genre: str | None = None,
        artist: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:

        tracks = list(
            self.library.values()
        )

        if genre:

            genre_value = genre.casefold()

            tracks = [
                track
                for track in tracks
                if genre_value
                in track.genre.casefold()
            ]

        if artist:

            artist_value = artist.casefold()

            tracks = [
                track
                for track in tracks
                if artist_value
                in track.artist.casefold()
            ]

        # Prefer tracks that have not
        # recently been played.
        recent = set(
            self.recently_played[-20:]
        )

        tracks.sort(
            key=lambda track:
            (
                track.track_id
                in recent,
                track.title.casefold(),
            )
        )

        return [
            track.to_dict()
            for track in tracks[:max(0, limit)]
        ]

    # ============================================================
    # SHUFFLE
    # ============================================================

    def shuffle_library(
        self,
    ) -> list[dict[str, Any]]:

        import random

        tracks = list(
            self.library.values()
        )

        random.shuffle(
            tracks
        )

        return [
            track.to_dict()
            for track in tracks
        ]

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        return {
            "library": self.get_library(),
            "favorites": list(
                self.favorites
            ),
            "recently_played": list(
                self.recently_played
            ),
            "search_history": list(
                self.search_history
            ),
        }

    def import_state(
        self,
        data: dict[str, Any],
    ) -> None:

        if not isinstance(data, dict):
            raise TypeError(
                "Music state must be a dictionary."
            )

        self.library.clear()

        for raw_track in data.get(
            "library",
            [],
        ):

            if not isinstance(
                raw_track,
                dict,
            ):
                continue

            track = Track(
                title=str(
                    raw_track.get(
                        "title",
                        "Unknown",
                    )
                ),
                artist=str(
                    raw_track.get(
                        "artist",
                        "Unknown Artist",
                    )
                ),
                album=str(
                    raw_track.get(
                        "album",
                        "",
                    )
                ),
                source=str(
                    raw_track.get(
                        "source",
                        "",
                    )
                ),
                duration=float(
                    raw_track.get(
                        "duration",
                        0.0,
                    )
                ),
                genre=str(
                    raw_track.get(
                        "genre",
                        "",
                    )
                ),
                year=raw_track.get(
                    "year"
                ),
                track_id=raw_track.get(
                    "track_id"
                ),
                metadata=dict(
                    raw_track.get(
                        "metadata",
                        {},
                    )
                ),
            )

            if track.track_id:
                self.library[
                    track.track_id
                ] = track

        self.favorites = {
            item
            for item in data.get(
                "favorites",
                [],
            )
            if item in self.library
        }

        self.recently_played = [
            item
            for item in data.get(
                "recently_played",
                [],
            )
            if item in self.library
        ][-100:]

        self.search_history = [
            str(item)
            for item in data.get(
                "search_history",
                [],
            )
        ][-100:]


__all__ = [
    "Track",
    "MusicManager",
]


