"""
RENIX Video Manager

Handles video library management, search, playback,
queue operations, favorites, recently watched videos,
and basic video metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .media_manager import MediaManager


@dataclass
class Video:
    """Represents a video."""

    title: str
    source: str
    duration: float = 0.0
    description: str = ""
    category: str = ""
    creator: str = ""
    year: int | None = None
    video_id: str | None = None
    thumbnail: str | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "source": self.source,
            "duration": self.duration,
            "description": self.description,
            "category": self.category,
            "creator": self.creator,
            "year": self.year,
            "video_id": self.video_id,
            "thumbnail": self.thumbnail,
            "metadata": dict(self.metadata),
        }


class VideoManager:
    """
    RENIX video subsystem.

    The manager handles video data and playback state.
    A real playback backend such as VLC, mpv, browser playback,
    or another player can be connected later.
    """

    def __init__(
        self,
        media_manager: MediaManager | None = None,
    ) -> None:

        self.media_manager = (
            media_manager
            or MediaManager()
        )

        self.library: dict[str, Video] = {}

        self.favorites: set[str] = set()

        self.recently_watched: list[str] = []

        self.watch_progress: dict[
            str,
            float,
        ] = {}

        self.search_history: list[str] = []

        self._video_counter = 0

    # ============================================================
    # LIBRARY
    # ============================================================

    def add_video(
        self,
        video: Video,
    ) -> str:

        if not isinstance(
            video,
            Video,
        ):
            raise TypeError(
                "video must be a Video instance."
            )

        if not video.video_id:

            self._video_counter += 1

            video.video_id = (
                f"VIDEO-{self._video_counter:06d}"
            )

        self.library[
            video.video_id
        ] = video

        return video.video_id

    def add_videos(
        self,
        videos: Iterable[Video],
    ) -> list[str]:

        ids = []

        for video in videos:
            ids.append(
                self.add_video(video)
            )

        return ids

    def get_video(
        self,
        video_id: str,
    ) -> Video | None:

        return self.library.get(
            video_id
        )

    def remove_video(
        self,
        video_id: str,
    ) -> bool:

        if video_id not in self.library:
            return False

        self.library.pop(
            video_id
        )

        self.favorites.discard(
            video_id
        )

        self.watch_progress.pop(
            video_id,
            None,
        )

        self.recently_watched = [
            item
            for item in self.recently_watched
            if item != video_id
        ]

        return True

    def get_library(
        self,
    ) -> list[dict[str, Any]]:

        return [
            video.to_dict()
            for video in self.library.values()
        ]

    def clear_library(self) -> None:

        self.library.clear()
        self.favorites.clear()
        self.recently_watched.clear()
        self.watch_progress.clear()

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

        for video in self.library.values():

            searchable = " ".join(
                [
                    video.title,
                    video.description,
                    video.category,
                    video.creator,
                    str(video.year or ""),
                ]
            ).casefold()

            if query in searchable:

                results.append(
                    video.to_dict()
                )

        return results[:max(0, limit)]

    def search_creator(
        self,
        creator: str,
    ) -> list[dict[str, Any]]:

        creator = (
            creator or ""
        ).strip().casefold()

        return [
            video.to_dict()
            for video in self.library.values()
            if creator
            in video.creator.casefold()
        ]

    def search_category(
        self,
        category: str,
    ) -> list[dict[str, Any]]:

        category = (
            category or ""
        ).strip().casefold()

        return [
            video.to_dict()
            for video in self.library.values()
            if category
            in video.category.casefold()
        ]

    # ============================================================
    # PLAYBACK
    # ============================================================

    def play(
        self,
        video: Video | str,
        *,
        start_position: float | None = None,
    ) -> dict[str, Any]:

        if isinstance(video, str):

            found = self.get_video(
                video
            )

            if found is None:

                return {
                    "success": False,
                    "message": (
                        f"Video '{video}' "
                        "was not found."
                    ),
                }

            video = found

        if not isinstance(
            video,
            Video,
        ):

            return {
                "success": False,
                "message": "Invalid video.",
            }

        if start_position is None:

            start_position = (
                self.watch_progress.get(
                    video.video_id or "",
                    0.0,
                )
            )

        start_position = max(
            0.0,
            float(start_position),
        )

        if video.duration > 0:

            start_position = min(
                start_position,
                video.duration,
            )

        result = self.media_manager.play(
            video.source,
            title=video.title,
            media_type="video",
            duration=video.duration,
        )

        self.media_manager.seek(
            start_position
        )

        if video.video_id:

            self.recently_watched.append(
                video.video_id
            )

            if len(
                self.recently_watched
            ) > 100:

                self.recently_watched = (
                    self.recently_watched[-100:]
                )

        return {
            "success": True,
            "video": video.to_dict(),
            "start_position": start_position,
            "media_status": self.media_manager.status(),
        }

    def pause(self) -> dict[str, Any]:

        self._save_current_progress()

        return self.media_manager.pause()

    def resume(self) -> dict[str, Any]:

        return self.media_manager.resume()

    def stop(self) -> dict[str, Any]:

        self._save_current_progress()

        return self.media_manager.stop()

    def seek(
        self,
        position: float,
    ) -> dict[str, Any]:

        result = self.media_manager.seek(
            position
        )

        self._save_current_progress()

        return result

    def forward(
        self,
        seconds: float = 10.0,
    ) -> dict[str, Any]:

        result = self.media_manager.forward(
            seconds
        )

        self._save_current_progress()

        return result

    def backward(
        self,
        seconds: float = 10.0,
    ) -> dict[str, Any]:

        result = self.media_manager.backward(
            seconds
        )

        self._save_current_progress()

        return result

    # ============================================================
    # WATCH PROGRESS
    # ============================================================

    def set_progress(
        self,
        video_id: str,
        position: float,
    ) -> bool:

        video = self.get_video(
            video_id
        )

        if video is None:
            return False

        position = max(
            0.0,
            float(position),
        )

        if video.duration > 0:

            position = min(
                position,
                video.duration,
            )

        self.watch_progress[
            video_id
        ] = position

        return True

    def get_progress(
        self,
        video_id: str,
    ) -> float:

        return self.watch_progress.get(
            video_id,
            0.0,
        )

    def get_progress_percent(
        self,
        video_id: str,
    ) -> float:

        video = self.get_video(
            video_id
        )

        if video is None:
            return 0.0

        if video.duration <= 0:
            return 0.0

        progress = self.get_progress(
            video_id
        )

        return min(
            100.0,
            max(
                0.0,
                (progress / video.duration)
                * 100.0,
            ),
        )

    def mark_watched(
        self,
        video_id: str,
    ) -> bool:

        video = self.get_video(
            video_id
        )

        if video is None:
            return False

        self.watch_progress[
            video_id
        ] = video.duration

        if video_id not in self.recently_watched:

            self.recently_watched.append(
                video_id
            )

        return True

    def _save_current_progress(self) -> None:

        current = (
            self.media_manager.current_media()
        )

        if not current:
            return

        if current.get(
            "media_type"
        ) != "video":
            return

        source = current.get(
            "source"
        )

        if not source:
            return

        for video in self.library.values():

            if video.source == source:

                if video.video_id:

                    self.set_progress(
                        video.video_id,
                        float(
                            current.get(
                                "position",
                                0.0,
                            )
                        ),
                    )

                break

    # ============================================================
    # FAVORITES
    # ============================================================

    def add_favorite(
        self,
        video_id: str,
    ) -> bool:

        if video_id not in self.library:
            return False

        self.favorites.add(
            video_id
        )

        return True

    def remove_favorite(
        self,
        video_id: str,
    ) -> bool:

        if video_id not in self.favorites:
            return False

        self.favorites.remove(
            video_id
        )

        return True

    def toggle_favorite(
        self,
        video_id: str,
    ) -> bool:

        if video_id in self.favorites:

            self.favorites.remove(
                video_id
            )

            return False

        if video_id in self.library:

            self.favorites.add(
                video_id
            )

            return True

        return False

    def is_favorite(
        self,
        video_id: str,
    ) -> bool:

        return video_id in self.favorites

    def get_favorites(
        self,
    ) -> list[dict[str, Any]]:

        return [
            self.library[
                video_id
            ].to_dict()
            for video_id in self.favorites
            if video_id in self.library
        ]

    # ============================================================
    # RECENTLY WATCHED
    # ============================================================

    def get_recently_watched(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:

        limit = max(
            0,
            int(limit),
        )

        results = []

        for video_id in reversed(
            self.recently_watched
        ):

            video = self.library.get(
                video_id
            )

            if video:

                data = video.to_dict()

                data["progress"] = (
                    self.get_progress(
                        video_id
                    )
                )

                data["progress_percent"] = (
                    self.get_progress_percent(
                        video_id
                    )
                )

                results.append(
                    data
                )

            if len(results) >= limit:
                break

        return results

    def clear_recently_watched(
        self,
    ) -> None:

        self.recently_watched.clear()

    # ============================================================
    # QUEUE
    # ============================================================

    def add_to_queue(
        self,
        video: Video | str,
    ) -> dict[str, Any]:

        if isinstance(video, str):
            video = self.get_video(video)

        if not isinstance(
            video,
            Video,
        ):

            return {
                "success": False,
                "message": "Video not found.",
            }

        result = self.media_manager.add_to_queue(
            video.source,
            title=video.title,
            media_type="video",
            duration=video.duration,
        )

        return {
            **result,
            "video": video.to_dict(),
        }

    def queue_videos(
        self,
        videos: Iterable[Video | str],
    ) -> list[dict[str, Any]]:

        return [
            self.add_to_queue(video)
            for video in videos
        ]

    def get_queue(
        self,
    ) -> list[dict[str, Any]]:

        return self.media_manager.get_queue()

    def clear_queue(self) -> dict[str, Any]:

        return self.media_manager.clear_queue()

    def play_next(self) -> dict[str, Any]:

        return self.media_manager.play_next()

    def play_previous(self) -> dict[str, Any]:

        return self.media_manager.play_previous()

    # ============================================================
    # RECOMMENDATIONS
    # ============================================================

    def recommend(
        self,
        *,
        category: str | None = None,
        creator: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:

        videos = list(
            self.library.values()
        )

        if category:

            category = category.casefold()

            videos = [
                video
                for video in videos
                if category
                in video.category.casefold()
            ]

        if creator:

            creator = creator.casefold()

            videos = [
                video
                for video in videos
                if creator
                in video.creator.casefold()
            ]

        recent = set(
            self.recently_watched[-20:]
        )

        videos.sort(
            key=lambda video: (
                video.video_id in recent,
                video.title.casefold(),
            )
        )

        return [
            video.to_dict()
            for video in videos[:max(0, limit)]
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
            "recently_watched": list(
                self.recently_watched
            ),
            "watch_progress": dict(
                self.watch_progress
            ),
            "search_history": list(
                self.search_history
            ),
        }

    def import_state(
        self,
        data: dict[str, Any],
    ) -> None:

        if not isinstance(
            data,
            dict,
        ):

            raise TypeError(
                "Video state must be a dictionary."
            )

        self.library.clear()

        for raw in data.get(
            "library",
            [],
        ):

            if not isinstance(
                raw,
                dict,
            ):
                continue

            video = Video(
                title=str(
                    raw.get(
                        "title",
                        "Unknown Video",
                    )
                ),
                source=str(
                    raw.get(
                        "source",
                        "",
                    )
                ),
                duration=float(
                    raw.get(
                        "duration",
                        0.0,
                    )
                ),
                description=str(
                    raw.get(
                        "description",
                        "",
                    )
                ),
                category=str(
                    raw.get(
                        "category",
                        "",
                    )
                ),
                creator=str(
                    raw.get(
                        "creator",
                        "",
                    )
                ),
                year=raw.get(
                    "year"
                ),
                video_id=raw.get(
                    "video_id"
                ),
                thumbnail=raw.get(
                    "thumbnail"
                ),
                metadata=dict(
                    raw.get(
                        "metadata",
                        {},
                    )
                ),
            )

            if video.video_id:
                self.library[
                    video.video_id
                ] = video

        self.favorites = {
            item
            for item in data.get(
                "favorites",
                [],
            )
            if item in self.library
        }

        self.recently_watched = [
            item
            for item in data.get(
                "recently_watched",
                [],
            )
            if item in self.library
        ][-100:]

        self.watch_progress = {
            str(key): max(
                0.0,
                float(value),
            )
            for key, value in data.get(
                "watch_progress",
                {},
            ).items()
            if str(key) in self.library
        }

        self.search_history = [
            str(item)
            for item in data.get(
                "search_history",
                [],
            )
        ][-100:]


__all__ = [
    "Video",
    "VideoManager",
]


