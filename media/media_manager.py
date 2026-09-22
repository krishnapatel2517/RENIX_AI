"""
RENIX Media Manager

Central controller for all media-related operations:
music, video, playlists, volume and casting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MediaState:
    """Current RENIX media state."""

    is_playing: bool = False
    is_paused: bool = False
    media_type: Optional[str] = None
    current_title: Optional[str] = None
    current_source: Optional[str] = None
    position: float = 0.0
    duration: float = 0.0
    volume: int = 70
    muted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_playing": self.is_playing,
            "is_paused": self.is_paused,
            "media_type": self.media_type,
            "current_title": self.current_title,
            "current_source": self.current_source,
            "position": self.position,
            "duration": self.duration,
            "volume": self.volume,
            "muted": self.muted,
        }


class MediaManager:
    """
    Central media controller for RENIX.

    Responsibilities:
        - Play and pause media
        - Stop media
        - Skip forward/backward
        - Manage volume
        - Track currently playing media
        - Maintain playback state
        - Coordinate music/video/playlists/casting
    """

    def __init__(
        self,
        *,
        default_volume: int = 70,
    ) -> None:

        self.state = MediaState(
            volume=self._clamp_volume(
                default_volume
            )
        )

        self.history: list[dict[str, Any]] = []

        self.queue: list[dict[str, Any]] = []

        self.current_index: int = -1

    # ============================================================
    # PLAYBACK
    # ============================================================

    def play(
        self,
        source: str,
        *,
        title: str | None = None,
        media_type: str = "unknown",
        duration: float = 0.0,
    ) -> dict[str, Any]:

        if not source:
            raise ValueError(
                "Media source cannot be empty."
            )

        self.state.is_playing = True
        self.state.is_paused = False
        self.state.media_type = media_type
        self.state.current_title = (
            title or source
        )
        self.state.current_source = source
        self.state.position = 0.0
        self.state.duration = max(
            0.0,
            duration,
        )

        self._add_history(
            action="play",
            source=source,
            title=title or source,
            media_type=media_type,
        )

        return self.status()

    def pause(self) -> dict[str, Any]:

        if not self.state.is_playing:
            return self.status()

        self.state.is_playing = False
        self.state.is_paused = True

        self._add_history(
            action="pause",
            source=self.state.current_source,
            title=self.state.current_title,
        )

        return self.status()

    def resume(self) -> dict[str, Any]:

        if (
            not self.state.is_paused
            and not self.state.current_source
        ):
            return self.status()

        self.state.is_playing = True
        self.state.is_paused = False

        self._add_history(
            action="resume",
            source=self.state.current_source,
            title=self.state.current_title,
        )

        return self.status()

    def stop(self) -> dict[str, Any]:

        self.state.is_playing = False
        self.state.is_paused = False
        self.state.position = 0.0

        self._add_history(
            action="stop",
            source=self.state.current_source,
            title=self.state.current_title,
        )

        return self.status()

    # ============================================================
    # SEEKING
    # ============================================================

    def seek(
        self,
        position: float,
    ) -> dict[str, Any]:

        position = max(
            0.0,
            float(position),
        )

        if self.state.duration > 0:
            position = min(
                position,
                self.state.duration,
            )

        self.state.position = position

        return self.status()

    def forward(
        self,
        seconds: float = 10.0,
    ) -> dict[str, Any]:

        return self.seek(
            self.state.position
            + max(0.0, seconds)
        )

    def backward(
        self,
        seconds: float = 10.0,
    ) -> dict[str, Any]:

        return self.seek(
            self.state.position
            - max(0.0, seconds)
        )

    # ============================================================
    # QUEUE
    # ============================================================

    def add_to_queue(
        self,
        source: str,
        *,
        title: str | None = None,
        media_type: str = "unknown",
        duration: float = 0.0,
    ) -> dict[str, Any]:

        item = {
            "source": source,
            "title": title or source,
            "media_type": media_type,
            "duration": max(
                0.0,
                float(duration),
            ),
        }

        self.queue.append(item)

        return {
            "success": True,
            "message": "Media added to queue.",
            "queue_size": len(self.queue),
            "item": item,
        }

    def remove_from_queue(
        self,
        index: int,
    ) -> dict[str, Any]:

        if not self.queue:
            return {
                "success": False,
                "message": "Queue is empty.",
            }

        if index < 0 or index >= len(self.queue):
            return {
                "success": False,
                "message": "Invalid queue index.",
            }

        item = self.queue.pop(index)

        return {
            "success": True,
            "message": "Media removed from queue.",
            "item": item,
            "queue_size": len(self.queue),
        }

    def clear_queue(self) -> dict[str, Any]:

        self.queue.clear()
        self.current_index = -1

        return {
            "success": True,
            "message": "Media queue cleared.",
        }

    def get_queue(self) -> list[dict[str, Any]]:

        return [
            dict(item)
            for item in self.queue
        ]

    def play_next(self) -> dict[str, Any]:

        if not self.queue:
            return {
                "success": False,
                "message": "Queue is empty.",
            }

        if self.current_index + 1 >= len(
            self.queue
        ):
            return {
                "success": False,
                "message": "No next media item.",
            }

        self.current_index += 1

        item = self.queue[
            self.current_index
        ]

        self.play(
            item["source"],
            title=item["title"],
            media_type=item["media_type"],
            duration=item["duration"],
        )

        return {
            "success": True,
            "item": item,
            "status": self.status(),
        }

    def play_previous(self) -> dict[str, Any]:

        if not self.queue:
            return {
                "success": False,
                "message": "Queue is empty.",
            }

        if self.current_index <= 0:
            return {
                "success": False,
                "message": "No previous media item.",
            }

        self.current_index -= 1

        item = self.queue[
            self.current_index
        ]

        self.play(
            item["source"],
            title=item["title"],
            media_type=item["media_type"],
            duration=item["duration"],
        )

        return {
            "success": True,
            "item": item,
            "status": self.status(),
        }

    # ============================================================
    # VOLUME
    # ============================================================

    def set_volume(
        self,
        volume: int,
    ) -> dict[str, Any]:

        self.state.volume = (
            self._clamp_volume(volume)
        )

        if self.state.volume > 0:
            self.state.muted = False

        return self.status()

    def increase_volume(
        self,
        amount: int = 10,
    ) -> dict[str, Any]:

        return self.set_volume(
            self.state.volume
            + max(0, amount)
        )

    def decrease_volume(
        self,
        amount: int = 10,
    ) -> dict[str, Any]:

        return self.set_volume(
            self.state.volume
            - max(0, amount)
        )

    def mute(self) -> dict[str, Any]:

        self.state.muted = True

        return self.status()

    def unmute(self) -> dict[str, Any]:

        self.state.muted = False

        return self.status()

    def toggle_mute(self) -> dict[str, Any]:

        self.state.muted = (
            not self.state.muted
        )

        return self.status()

    # ============================================================
    # STATUS
    # ============================================================

    def status(self) -> dict[str, Any]:

        return {
            "success": True,
            "media": self.state.to_dict(),
            "queue_size": len(self.queue),
            "current_queue_index": (
                self.current_index
            ),
        }

    def is_active(self) -> bool:

        return (
            self.state.is_playing
            or self.state.is_paused
        )

    def current_media(
        self,
    ) -> dict[str, Any] | None:

        if not self.state.current_source:
            return None

        return {
            "title": self.state.current_title,
            "source": self.state.current_source,
            "media_type": self.state.media_type,
            "position": self.state.position,
            "duration": self.state.duration,
        }

    # ============================================================
    # HISTORY
    # ============================================================

    def _add_history(
        self,
        *,
        action: str,
        source: str | None,
        title: str | None,
        media_type: str | None = None,
    ) -> None:

        self.history.append(
            {
                "action": action,
                "source": source,
                "title": title,
                "media_type": media_type,
            }
        )

        # Prevent unlimited in-memory growth.
        if len(self.history) > 500:
            self.history = self.history[-500:]

    def get_history(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:

        limit = max(0, int(limit))

        if limit == 0:
            return []

        return [
            dict(item)
            for item in self.history[-limit:]
        ]

    def clear_history(self) -> None:

        self.history.clear()

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def export_state(self) -> dict[str, Any]:

        return {
            "state": self.state.to_dict(),
            "queue": self.get_queue(),
            "current_index": self.current_index,
            "history": self.get_history(500),
        }

    def import_state(
        self,
        data: dict[str, Any],
    ) -> None:

        if not isinstance(data, dict):
            raise TypeError(
                "Media state must be a dictionary."
            )

        state = data.get(
            "state",
            {},
        )

        if isinstance(state, dict):

            self.state = MediaState(
                is_playing=bool(
                    state.get(
                        "is_playing",
                        False,
                    )
                ),
                is_paused=bool(
                    state.get(
                        "is_paused",
                        False,
                    )
                ),
                media_type=state.get(
                    "media_type"
                ),
                current_title=state.get(
                    "current_title"
                ),
                current_source=state.get(
                    "current_source"
                ),
                position=float(
                    state.get(
                        "position",
                        0.0,
                    )
                ),
                duration=float(
                    state.get(
                        "duration",
                        0.0,
                    )
                ),
                volume=self._clamp_volume(
                    state.get(
                        "volume",
                        70,
                    )
                ),
                muted=bool(
                    state.get(
                        "muted",
                        False,
                    )
                ),
            )

        queue = data.get(
            "queue",
            [],
        )

        if isinstance(queue, list):
            self.queue = [
                item
                for item in queue
                if isinstance(item, dict)
            ]

        self.current_index = int(
            data.get(
                "current_index",
                -1,
            )
        )

        history = data.get(
            "history",
            [],
        )

        if isinstance(history, list):
            self.history = [
                item
                for item in history
                if isinstance(item, dict)
            ][-500:]

    # ============================================================
    # UTILITIES
    # ============================================================

    @staticmethod
    def _clamp_volume(
        value: int | float,
    ) -> int:

        try:
            value = int(value)
        except (
            TypeError,
            ValueError,
        ):
            value = 70

        return max(
            0,
            min(100, value),
        )


__all__ = [
    "MediaState",
    "MediaManager",
]


