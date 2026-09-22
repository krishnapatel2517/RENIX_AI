"""
RENIX Media Service
===================

High-level media orchestration layer.

MediaService sits above individual providers such as Spotify, YouTube,
local media players, or other supported services.

The rest of RENIX can therefore use one consistent interface for:

    - play / pause / stop
    - next / previous
    - search
    - volume
    - queue
    - playlists
    - shuffle / repeat
    - playback devices
    - now playing
"""

from __future__ import annotations

from typing import Any, Iterable

from .media_provider import MediaProvider


class MediaService:
    """
    High-level manager for RENIX media providers.
    """

    def __init__(
        self,
        providers: Iterable[MediaProvider] | None = None,
        *,
        default_provider: str | None = None,
    ) -> None:
        self._providers: dict[str, MediaProvider] = {}

        for provider in providers or []:
            self.register_provider(
                provider
            )

        self._active_provider_name = (
            default_provider
        )

        if (
            self._active_provider_name
            and self._active_provider_name
            not in self._providers
        ):
            raise ValueError(
                f"Unknown default media provider: "
                f"{self._active_provider_name}"
            )

    # ========================================================
    # PROVIDERS
    # ========================================================

    def register_provider(
        self,
        provider: MediaProvider,
        *,
        replace: bool = False,
    ) -> None:
        """Register a media provider."""

        if not isinstance(
            provider,
            MediaProvider,
        ):
            raise TypeError(
                "provider must inherit from MediaProvider."
            )

        name = provider.name.strip()

        if not name:
            raise ValueError(
                "Media provider must have a name."
            )

        if (
            name in self._providers
            and not replace
        ):
            raise ValueError(
                f"Media provider '{name}' is already registered."
            )

        self._providers[name] = provider

        if self._active_provider_name is None:
            self._active_provider_name = name

    def unregister_provider(
        self,
        name: str,
    ) -> bool:
        """Remove a provider."""

        if name not in self._providers:
            return False

        del self._providers[name]

        if self._active_provider_name == name:
            self._active_provider_name = (
                next(
                    iter(
                        self._providers
                    ),
                    None,
                )
            )

        return True

    def get_provider(
        self,
        name: str | None = None,
    ) -> MediaProvider:
        """Return a provider."""

        provider_name = (
            name
            or self._active_provider_name
        )

        if not provider_name:
            raise RuntimeError(
                "No media provider is configured."
            )

        provider = self._providers.get(
            provider_name
        )

        if provider is None:
            raise KeyError(
                f"Unknown media provider: "
                f"{provider_name}"
            )

        return provider

    def set_provider(
        self,
        name: str,
    ) -> MediaProvider:
        """Set the active media provider."""

        provider = self.get_provider(
            name
        )

        self._active_provider_name = name

        return provider

    def active_provider(
        self,
    ) -> MediaProvider:
        """Return the active provider."""

        return self.get_provider()

    def provider_names(
        self,
    ) -> list[str]:
        """Return registered provider names."""

        return list(
            self._providers.keys()
        )

    # ========================================================
    # PLAYBACK
    # ========================================================

    def play(
        self,
        media_id: str | None = None,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Start or resume playback."""

        return self.get_provider(
            provider
        ).play(
            media_id=media_id,
            **kwargs,
        )

    def pause(
        self,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Pause playback."""

        return self.get_provider(
            provider
        ).pause(
            **kwargs
        )

    def stop(
        self,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stop playback."""

        return self.get_provider(
            provider
        ).stop(
            **kwargs
        )

    def next(
        self,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Play next item."""

        return self.get_provider(
            provider
        ).next(
            **kwargs
        )

    def previous(
        self,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Play previous item."""

        return self.get_provider(
            provider
        ).previous(
            **kwargs
        )

    # ========================================================
    # NOW PLAYING
    # ========================================================

    def now_playing(
        self,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Return currently playing media."""

        return self.get_provider(
            provider
        ).now_playing(
            **kwargs
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        *,
        media_type: str | None = None,
        limit: int = 20,
        provider: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search media using the active provider."""

        if not query.strip():
            return []

        return self.get_provider(
            provider
        ).search(
            query,
            media_type=media_type,
            limit=limit,
            **kwargs,
        )

    # ========================================================
    # VOLUME
    # ========================================================

    def get_volume(
        self,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> float:
        """Get current volume."""

        return self.get_provider(
            provider
        ).get_volume(
            **kwargs
        )

    def set_volume(
        self,
        volume: float,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Set volume."""

        if not 0.0 <= volume <= 1.0:
            raise ValueError(
                "volume must be between 0.0 and 1.0."
            )

        return self.get_provider(
            provider
        ).set_volume(
            volume,
            **kwargs,
        )

    def volume_up(
        self,
        amount: float = 0.1,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Increase volume."""

        if amount < 0:
            raise ValueError(
                "amount cannot be negative."
            )

        return self.get_provider(
            provider
        ).volume_up(
            amount,
            **kwargs,
        )

    def volume_down(
        self,
        amount: float = 0.1,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Decrease volume."""

        if amount < 0:
            raise ValueError(
                "amount cannot be negative."
            )

        return self.get_provider(
            provider
        ).volume_down(
            amount,
            **kwargs,
        )

    # ========================================================
    # PLAYLISTS
    # ========================================================

    def list_playlists(
        self,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """List playlists."""

        return self.get_provider(
            provider
        ).list_playlists(
            **kwargs
        )

    def get_playlist(
        self,
        playlist_id: str,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get a playlist."""

        return self.get_provider(
            provider
        ).get_playlist(
            playlist_id,
            **kwargs,
        )

    def create_playlist(
        self,
        name: str,
        *,
        description: str = "",
        provider: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a playlist."""

        if not name.strip():
            raise ValueError(
                "Playlist name cannot be empty."
            )

        return self.get_provider(
            provider
        ).create_playlist(
            name,
            description=description,
            **kwargs,
        )

    def add_to_playlist(
        self,
        playlist_id: str,
        media_ids: list[str],
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Add media to a playlist."""

        if not media_ids:
            return None

        return self.get_provider(
            provider
        ).add_to_playlist(
            playlist_id,
            media_ids,
            **kwargs,
        )

    def remove_from_playlist(
        self,
        playlist_id: str,
        media_ids: list[str],
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Remove media from a playlist."""

        if not media_ids:
            return None

        return self.get_provider(
            provider
        ).remove_from_playlist(
            playlist_id,
            media_ids,
            **kwargs,
        )

    # ========================================================
    # QUEUE
    # ========================================================

    def get_queue(
        self,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return playback queue."""

        return self.get_provider(
            provider
        ).get_queue(
            **kwargs
        )

    def add_to_queue(
        self,
        media_id: str,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Add an item to the queue."""

        return self.get_provider(
            provider
        ).add_to_queue(
            media_id,
            **kwargs,
        )

    def clear_queue(
        self,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Clear playback queue."""

        return self.get_provider(
            provider
        ).clear_queue(
            **kwargs
        )

    # ========================================================
    # SEEK
    # ========================================================

    def get_position(
        self,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> float:
        """Return playback position."""

        return self.get_provider(
            provider
        ).get_position(
            **kwargs
        )

    def seek(
        self,
        position: float,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Seek playback."""

        if position < 0:
            raise ValueError(
                "position cannot be negative."
            )

        return self.get_provider(
            provider
        ).seek(
            position,
            **kwargs,
        )

    # ========================================================
    # SHUFFLE / REPEAT
    # ========================================================

    def set_shuffle(
        self,
        enabled: bool,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Enable or disable shuffle."""

        return self.get_provider(
            provider
        ).set_shuffle(
            enabled,
            **kwargs,
        )

    def set_repeat(
        self,
        mode: str,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Set repeat mode."""

        allowed = {
            "off",
            "one",
            "all",
        }

        normalized = mode.lower().strip()

        if normalized not in allowed:
            raise ValueError(
                "repeat mode must be one of: "
                "off, one, all."
            )

        return self.get_provider(
            provider
        ).set_repeat(
            normalized,
            **kwargs,
        )

    # ========================================================
    # DEVICES
    # ========================================================

    def list_devices(
        self,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """List playback devices."""

        return self.get_provider(
            provider
        ).list_devices(
            **kwargs
        )

    def set_device(
        self,
        device_id: str,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Select playback device."""

        return self.get_provider(
            provider
        ).set_device(
            device_id,
            **kwargs,
        )

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(
        self,
        *,
        provider: str | None = None,
    ) -> bool:
        """Check provider health."""

        return self.get_provider(
            provider
        ).health_check()

    def health_all(
        self,
    ) -> dict[str, bool]:
        """Check all registered providers."""

        return {
            name: provider.health_check()
            for name, provider
            in self._providers.items()
        }

    # ========================================================
    # CAPABILITIES
    # ========================================================

    def capabilities(
        self,
        *,
        provider: str | None = None,
    ) -> dict[str, bool]:
        """Return active provider capabilities."""

        return self.get_provider(
            provider
        ).capabilities()

    def capabilities_all(
        self,
    ) -> dict[str, dict[str, bool]]:
        """Return capabilities for all providers."""

        return {
            name: provider.capabilities()
            for name, provider
            in self._providers.items()
        }

    # ========================================================
    # INFO
    # ========================================================

    def info(
        self,
        *,
        provider: str | None = None,
    ) -> dict[str, Any]:
        """Return active provider information."""

        selected = self.get_provider(
            provider
        )

        return {
            "active_provider": selected.name,
            "providers": self.provider_names(),
            "capabilities": selected.capabilities(),
            "healthy": selected.health_check(),
        }


__all__ = [
    "MediaService",
]


