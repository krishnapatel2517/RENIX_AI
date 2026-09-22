"""
RENIX Media Provider
====================

Base interface for media providers used by RENIX.

A media provider can control music, video, playlists, playback state,
volume, and media discovery without the rest of RENIX needing to know
which service is being used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MediaProvider(ABC):
    """
    Abstract base class for all RENIX media providers.
    """

    name: str = "unknown"

    # ========================================================
    # PLAYBACK
    # ========================================================

    @abstractmethod
    def play(
        self,
        *,
        media_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Start or resume media playback."""
        raise NotImplementedError

    @abstractmethod
    def pause(
        self,
        **kwargs: Any,
    ) -> Any:
        """Pause current playback."""
        raise NotImplementedError

    @abstractmethod
    def stop(
        self,
        **kwargs: Any,
    ) -> Any:
        """Stop current playback."""
        raise NotImplementedError

    @abstractmethod
    def next(
        self,
        **kwargs: Any,
    ) -> Any:
        """Play the next item."""
        raise NotImplementedError

    @abstractmethod
    def previous(
        self,
        **kwargs: Any,
    ) -> Any:
        """Play the previous item."""
        raise NotImplementedError

    # ========================================================
    # VOLUME
    # ========================================================

    @abstractmethod
    def get_volume(
        self,
        **kwargs: Any,
    ) -> float:
        """Return volume from 0.0 to 1.0."""
        raise NotImplementedError

    @abstractmethod
    def set_volume(
        self,
        volume: float,
        **kwargs: Any,
    ) -> Any:
        """Set volume from 0.0 to 1.0."""
        raise NotImplementedError

    def volume_up(
        self,
        amount: float = 0.1,
        **kwargs: Any,
    ) -> Any:
        """Increase volume."""

        current = self.get_volume(
            **kwargs
        )

        return self.set_volume(
            min(
                1.0,
                current + amount,
            ),
            **kwargs,
        )

    def volume_down(
        self,
        amount: float = 0.1,
        **kwargs: Any,
    ) -> Any:
        """Decrease volume."""

        current = self.get_volume(
            **kwargs
        )

        return self.set_volume(
            max(
                0.0,
                current - amount,
            ),
            **kwargs,
        )

    # ========================================================
    # NOW PLAYING
    # ========================================================

    @abstractmethod
    def now_playing(
        self,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Return information about currently playing media."""
        raise NotImplementedError

    # ========================================================
    # SEARCH
    # ========================================================

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        media_type: str | None = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search for media."""
        raise NotImplementedError

    # ========================================================
    # PLAYLISTS
    # ========================================================

    @abstractmethod
    def list_playlists(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return available playlists."""
        raise NotImplementedError

    @abstractmethod
    def get_playlist(
        self,
        playlist_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a playlist."""
        raise NotImplementedError

    # ========================================================
    # PLAYLIST MANAGEMENT
    # ========================================================

    def create_playlist(
        self,
        name: str,
        *,
        description: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a playlist if supported."""

        raise NotImplementedError(
            f"{self.name} does not support playlist creation."
        )

    def add_to_playlist(
        self,
        playlist_id: str,
        media_ids: list[str],
        **kwargs: Any,
    ) -> Any:
        """Add media items to a playlist."""

        raise NotImplementedError(
            f"{self.name} does not support playlist editing."
        )

    def remove_from_playlist(
        self,
        playlist_id: str,
        media_ids: list[str],
        **kwargs: Any,
    ) -> Any:
        """Remove media items from a playlist."""

        raise NotImplementedError(
            f"{self.name} does not support playlist editing."
        )

    # ========================================================
    # QUEUE
    # ========================================================

    def get_queue(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return the current playback queue."""

        raise NotImplementedError(
            f"{self.name} does not support queue access."
        )

    def add_to_queue(
        self,
        media_id: str,
        **kwargs: Any,
    ) -> Any:
        """Add an item to the playback queue."""

        raise NotImplementedError(
            f"{self.name} does not support queue management."
        )

    def clear_queue(
        self,
        **kwargs: Any,
    ) -> Any:
        """Clear the playback queue."""

        raise NotImplementedError(
            f"{self.name} does not support queue management."
        )

    # ========================================================
    # SEEK
    # ========================================================

    def get_position(
        self,
        **kwargs: Any,
    ) -> float:
        """Return playback position in seconds."""

        raise NotImplementedError(
            f"{self.name} does not support playback position."
        )

    def seek(
        self,
        position: float,
        **kwargs: Any,
    ) -> Any:
        """Seek to a position in seconds."""

        raise NotImplementedError(
            f"{self.name} does not support seeking."
        )

    # ========================================================
    # REPEAT / SHUFFLE
    # ========================================================

    def set_shuffle(
        self,
        enabled: bool,
        **kwargs: Any,
    ) -> Any:
        """Enable or disable shuffle."""

        raise NotImplementedError(
            f"{self.name} does not support shuffle."
        )

    def set_repeat(
        self,
        mode: str,
        **kwargs: Any,
    ) -> Any:
        """
        Set repeat mode.

        Expected values can include:
            off
            one
            all
        """

        raise NotImplementedError(
            f"{self.name} does not support repeat mode."
        )

    # ========================================================
    # DEVICE
    # ========================================================

    def list_devices(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return available playback devices."""

        raise NotImplementedError(
            f"{self.name} does not support device discovery."
        )

    def set_device(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Select a playback device."""

        raise NotImplementedError(
            f"{self.name} does not support device selection."
        )

    # ========================================================
    # CAPABILITIES
    # ========================================================

    def capabilities(
        self,
    ) -> dict[str, bool]:
        """Return provider capabilities."""

        return {
            "playback": True,
            "pause": True,
            "stop": True,
            "next": True,
            "previous": True,
            "volume": True,
            "search": True,
            "playlists": True,
            "playlist_editing": False,
            "queue": False,
            "seek": False,
            "shuffle": False,
            "repeat": False,
            "devices": False,
        }

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(
        self,
    ) -> bool:
        """Return whether the provider is operational."""

        return True

    # ========================================================
    # INFORMATION
    # ========================================================

    def info(
        self,
    ) -> dict[str, Any]:
        """Return provider metadata."""

        return {
            "name": self.name,
            "capabilities": self.capabilities(),
        }


__all__ = [
    "MediaProvider",
]


