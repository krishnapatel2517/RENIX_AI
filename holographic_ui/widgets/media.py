"""
RENIX Media Widget
"""

from __future__ import annotations

from typing import Any


class MediaWidget:

    def __init__(self) -> None:

        self.visible = True

        self.playing = False

        self.title = ""

        self.artist = ""

        self.album = ""

        self.position = 0.0

        self.duration = 0.0

        self.volume = 1.0

    def set_track(
        self,
        title: str,
        artist: str = "",
        album: str = "",
        duration: float = 0.0,
    ) -> None:

        self.title = title
        self.artist = artist
        self.album = album
        self.duration = duration
        self.position = 0.0

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def stop(self) -> None:

        self.playing = False
        self.position = 0.0

    def set_volume(
        self,
        volume: float,
    ) -> None:

        self.volume = max(
            0.0,
            min(1.0, volume),
        )

    def seek(
        self,
        position: float,
    ) -> None:

        self.position = max(
            0.0,
            min(
                self.duration,
                position,
            ),
        )

    def get_data(self) -> dict[str, Any]:

        return {
            "playing": self.playing,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "position": self.position,
            "duration": self.duration,
            "volume": self.volume,
        }


