"""
RENIX Playlist Manager

Manages music/video playlists, ordering, searching,
shuffle, repeat modes, favorites, and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Iterable


@dataclass
class PlaylistItem:
    """A single item inside a playlist."""

    source: str
    title: str
    media_type: str = "unknown"
    duration: float = 0.0
    item_id: str | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "media_type": self.media_type,
            "duration": self.duration,
            "item_id": self.item_id,
            "metadata": dict(self.metadata),
        }


@dataclass
class Playlist:
    """Represents a RENIX playlist."""

    playlist_id: str
    name: str
    description: str = ""
    items: list[PlaylistItem] = field(
        default_factory=list
    )
    favorite: bool = False
    shuffle: bool = False
    repeat_mode: str = "off"
    current_index: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "playlist_id": self.playlist_id,
            "name": self.name,
            "description": self.description,
            "items": [
                item.to_dict()
                for item in self.items
            ],
            "favorite": self.favorite,
            "shuffle": self.shuffle,
            "repeat_mode": self.repeat_mode,
            "current_index": self.current_index,
        }


class PlaylistManager:
    """
    Central playlist manager.

    Supported repeat modes:

        off
        one
        all
    """

    VALID_REPEAT_MODES = {
        "off",
        "one",
        "all",
    }

    def __init__(self) -> None:

        self.playlists: dict[
            str,
            Playlist,
        ] = {}

        self.active_playlist_id: str | None = None

        self._playlist_counter = 0
        self._item_counter = 0

    # ============================================================
    # PLAYLIST CREATION
    # ============================================================

    def create_playlist(
        self,
        name: str,
        *,
        description: str = "",
        playlist_id: str | None = None,
    ) -> Playlist:

        name = (
            name or ""
        ).strip()

        if not name:
            raise ValueError(
                "Playlist name cannot be empty."
            )

        if playlist_id is None:

            self._playlist_counter += 1

            playlist_id = (
                f"PLAYLIST-{self._playlist_counter:06d}"
            )

        if playlist_id in self.playlists:
            raise ValueError(
                f"Playlist '{playlist_id}' already exists."
            )

        playlist = Playlist(
            playlist_id=playlist_id,
            name=name,
            description=description,
        )

        self.playlists[
            playlist_id
        ] = playlist

        return playlist

    def delete_playlist(
        self,
        playlist_id: str,
    ) -> bool:

        if playlist_id not in self.playlists:
            return False

        self.playlists.pop(
            playlist_id
        )

        if self.active_playlist_id == playlist_id:
            self.active_playlist_id = None

        return True

    def rename_playlist(
        self,
        playlist_id: str,
        new_name: str,
    ) -> bool:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None:
            return False

        new_name = (
            new_name or ""
        ).strip()

        if not new_name:
            return False

        playlist.name = new_name

        return True

    def get_playlist(
        self,
        playlist_id: str,
    ) -> Playlist | None:

        return self.playlists.get(
            playlist_id
        )

    def get_all_playlists(
        self,
    ) -> list[dict[str, Any]]:

        return [
            playlist.to_dict()
            for playlist in self.playlists.values()
        ]

    # ============================================================
    # ACTIVE PLAYLIST
    # ============================================================

    def set_active_playlist(
        self,
        playlist_id: str,
    ) -> bool:

        if playlist_id not in self.playlists:
            return False

        self.active_playlist_id = playlist_id

        return True

    def get_active_playlist(
        self,
    ) -> Playlist | None:

        if self.active_playlist_id is None:
            return None

        return self.playlists.get(
            self.active_playlist_id
        )

    # ============================================================
    # ITEMS
    # ============================================================

    def add_item(
        self,
        playlist_id: str,
        source: str,
        *,
        title: str | None = None,
        media_type: str = "unknown",
        duration: float = 0.0,
        item_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PlaylistItem | None:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None:
            return None

        if not source:
            raise ValueError(
                "Playlist item source cannot be empty."
            )

        if item_id is None:

            self._item_counter += 1

            item_id = (
                f"ITEM-{self._item_counter:06d}"
            )

        item = PlaylistItem(
            source=source,
            title=title or source,
            media_type=media_type,
            duration=max(
                0.0,
                float(duration),
            ),
            item_id=item_id,
            metadata=dict(
                metadata or {}
            ),
        )

        playlist.items.append(
            item
        )

        return item

    def add_items(
        self,
        playlist_id: str,
        items: Iterable[
            PlaylistItem | dict[str, Any]
        ],
    ) -> list[PlaylistItem]:

        added = []

        for item in items:

            if isinstance(
                item,
                PlaylistItem,
            ):

                result = self.add_item(
                    playlist_id,
                    item.source,
                    title=item.title,
                    media_type=item.media_type,
                    duration=item.duration,
                    item_id=item.item_id,
                    metadata=item.metadata,
                )

            elif isinstance(
                item,
                dict,
            ):

                result = self.add_item(
                    playlist_id,
                    str(
                        item.get(
                            "source",
                            "",
                        )
                    ),
                    title=item.get(
                        "title"
                    ),
                    media_type=str(
                        item.get(
                            "media_type",
                            "unknown",
                        )
                    ),
                    duration=float(
                        item.get(
                            "duration",
                            0.0,
                        )
                    ),
                    item_id=item.get(
                        "item_id"
                    ),
                    metadata=item.get(
                        "metadata",
                        {},
                    ),
                )

            else:
                continue

            if result:
                added.append(result)

        return added

    def remove_item(
        self,
        playlist_id: str,
        index: int,
    ) -> bool:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None:
            return False

        if index < 0 or index >= len(
            playlist.items
        ):
            return False

        playlist.items.pop(
            index
        )

        if playlist.current_index >= len(
            playlist.items
        ):
            playlist.current_index = (
                len(playlist.items) - 1
            )

        return True

    def remove_item_by_id(
        self,
        playlist_id: str,
        item_id: str,
    ) -> bool:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None:
            return False

        for index, item in enumerate(
            playlist.items
        ):

            if item.item_id == item_id:

                return self.remove_item(
                    playlist_id,
                    index,
                )

        return False

    def clear_playlist(
        self,
        playlist_id: str,
    ) -> bool:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None:
            return False

        playlist.items.clear()
        playlist.current_index = -1

        return True

    # ============================================================
    # ITEM ORDER
    # ============================================================

    def move_item(
        self,
        playlist_id: str,
        old_index: int,
        new_index: int,
    ) -> bool:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None:
            return False

        count = len(
            playlist.items
        )

        if (
            old_index < 0
            or old_index >= count
            or new_index < 0
            or new_index >= count
        ):
            return False

        item = playlist.items.pop(
            old_index
        )

        playlist.items.insert(
            new_index,
            item,
        )

        if playlist.current_index == old_index:
            playlist.current_index = (
                new_index
            )

        return True

    def shuffle_playlist(
        self,
        playlist_id: str,
    ) -> bool:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None:
            return False

        random.shuffle(
            playlist.items
        )

        playlist.shuffle = True

        if playlist.items:
            playlist.current_index = 0
        else:
            playlist.current_index = -1

        return True

    def unshuffle_playlist(
        self,
        playlist_id: str,
    ) -> bool:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None:
            return False

        playlist.shuffle = False

        return True

    # ============================================================
    # NAVIGATION
    # ============================================================

    def first(
        self,
        playlist_id: str,
    ) -> PlaylistItem | None:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None or not playlist.items:
            return None

        playlist.current_index = 0

        return playlist.items[0]

    def current(
        self,
        playlist_id: str,
    ) -> PlaylistItem | None:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None:
            return None

        index = playlist.current_index

        if index < 0 or index >= len(
            playlist.items
        ):
            return None

        return playlist.items[index]

    def next(
        self,
        playlist_id: str,
    ) -> PlaylistItem | None:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None or not playlist.items:
            return None

        if playlist.repeat_mode == "one":

            return self.current(
                playlist_id
            )

        next_index = (
            playlist.current_index + 1
        )

        if next_index >= len(
            playlist.items
        ):

            if playlist.repeat_mode == "all":
                next_index = 0
            else:
                return None

        playlist.current_index = (
            next_index
        )

        return playlist.items[
            next_index
        ]

    def previous(
        self,
        playlist_id: str,
    ) -> PlaylistItem | None:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None or not playlist.items:
            return None

        previous_index = (
            playlist.current_index - 1
        )

        if previous_index < 0:

            if playlist.repeat_mode == "all":
                previous_index = (
                    len(playlist.items) - 1
                )
            else:
                return None

        playlist.current_index = (
            previous_index
        )

        return playlist.items[
            previous_index
        ]

    # ============================================================
    # REPEAT / SHUFFLE
    # ============================================================

    def set_repeat_mode(
        self,
        playlist_id: str,
        mode: str,
    ) -> bool:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None:
            return False

        mode = (
            mode or ""
        ).strip().lower()

        if mode not in self.VALID_REPEAT_MODES:
            raise ValueError(
                "Repeat mode must be "
                "'off', 'one', or 'all'."
            )

        playlist.repeat_mode = mode

        return True

    def toggle_repeat(
        self,
        playlist_id: str,
    ) -> str | None:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None:
            return None

        order = [
            "off",
            "all",
            "one",
        ]

        current = playlist.repeat_mode

        next_index = (
            order.index(current) + 1
        ) % len(order)

        playlist.repeat_mode = (
            order[next_index]
        )

        return playlist.repeat_mode

    def toggle_shuffle(
        self,
        playlist_id: str,
    ) -> bool | None:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None:
            return None

        playlist.shuffle = (
            not playlist.shuffle
        )

        if playlist.shuffle:
            random.shuffle(
                playlist.items
            )

        return playlist.shuffle

    # ============================================================
    # FAVORITES
    # ============================================================

    def toggle_favorite(
        self,
        playlist_id: str,
    ) -> bool | None:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None:
            return None

        playlist.favorite = (
            not playlist.favorite
        )

        return playlist.favorite

    def get_favorite_playlists(
        self,
    ) -> list[dict[str, Any]]:

        return [
            playlist.to_dict()
            for playlist in self.playlists.values()
            if playlist.favorite
        ]

    # ============================================================
    # SEARCH
    # ============================================================

    def search_playlists(
        self,
        query: str,
    ) -> list[dict[str, Any]]:

        query = (
            query or ""
        ).strip().casefold()

        if not query:
            return []

        return [
            playlist.to_dict()
            for playlist in self.playlists.values()
            if (
                query
                in playlist.name.casefold()
                or query
                in playlist.description.casefold()
            )
        ]

    # ============================================================
    # DUPLICATION
    # ============================================================

    def duplicate_playlist(
        self,
        playlist_id: str,
        new_name: str | None = None,
    ) -> Playlist | None:

        original = self.get_playlist(
            playlist_id
        )

        if original is None:
            return None

        name = (
            new_name
            or f"{original.name} Copy"
        )

        copied = self.create_playlist(
            name,
            description=original.description,
        )

        copied.items = [
            PlaylistItem(
                source=item.source,
                title=item.title,
                media_type=item.media_type,
                duration=item.duration,
                metadata=dict(
                    item.metadata
                ),
            )
            for item in original.items
        ]

        copied.shuffle = original.shuffle
        copied.repeat_mode = (
            original.repeat_mode
        )

        return copied

    # ============================================================
    # STATISTICS
    # ============================================================

    def playlist_stats(
        self,
        playlist_id: str,
    ) -> dict[str, Any] | None:

        playlist = self.get_playlist(
            playlist_id
        )

        if playlist is None:
            return None

        total_duration = sum(
            item.duration
            for item in playlist.items
        )

        media_types: dict[
            str,
            int,
        ] = {}

        for item in playlist.items:

            media_types[
                item.media_type
            ] = (
                media_types.get(
                    item.media_type,
                    0,
                )
                + 1
            )

        return {
            "playlist_id": playlist.playlist_id,
            "name": playlist.name,
            "item_count": len(
                playlist.items
            ),
            "total_duration": (
                total_duration
            ),
            "media_types": media_types,
            "favorite": playlist.favorite,
            "shuffle": playlist.shuffle,
            "repeat_mode": (
                playlist.repeat_mode
            ),
        }

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        return {
            "playlists": [
                playlist.to_dict()
                for playlist in self.playlists.values()
            ],
            "active_playlist_id": (
                self.active_playlist_id
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
                "Playlist state must be a dictionary."
            )

        self.playlists.clear()

        raw_playlists = data.get(
            "playlists",
            [],
        )

        if not isinstance(
            raw_playlists,
            list,
        ):
            raw_playlists = []

        for raw_playlist in raw_playlists:

            if not isinstance(
                raw_playlist,
                dict,
            ):
                continue

            playlist_id = str(
                raw_playlist.get(
                    "playlist_id",
                    "",
                )
            )

            name = str(
                raw_playlist.get(
                    "name",
                    "Untitled Playlist",
                )
            )

            if not playlist_id:
                continue

            playlist = Playlist(
                playlist_id=playlist_id,
                name=name,
                description=str(
                    raw_playlist.get(
                        "description",
                        "",
                    )
                ),
                favorite=bool(
                    raw_playlist.get(
                        "favorite",
                        False,
                    )
                ),
                shuffle=bool(
                    raw_playlist.get(
                        "shuffle",
                        False,
                    )
                ),
                repeat_mode=str(
                    raw_playlist.get(
                        "repeat_mode",
                        "off",
                    )
                ),
                current_index=int(
                    raw_playlist.get(
                        "current_index",
                        -1,
                    )
                ),
            )

            if (
                playlist.repeat_mode
                not in self.VALID_REPEAT_MODES
            ):
                playlist.repeat_mode = "off"

            raw_items = raw_playlist.get(
                "items",
                [],
            )

            if isinstance(
                raw_items,
                list,
            ):

                for raw_item in raw_items:

                    if not isinstance(
                        raw_item,
                        dict,
                    ):
                        continue

                    source = str(
                        raw_item.get(
                            "source",
                            "",
                        )
                    )

                    if not source:
                        continue

                    playlist.items.append(
                        PlaylistItem(
                            source=source,
                            title=str(
                                raw_item.get(
                                    "title",
                                    source,
                                )
                            ),
                            media_type=str(
                                raw_item.get(
                                    "media_type",
                                    "unknown",
                                )
                            ),
                            duration=float(
                                raw_item.get(
                                    "duration",
                                    0.0,
                                )
                            ),
                            item_id=raw_item.get(
                                "item_id"
                            ),
                            metadata=dict(
                                raw_item.get(
                                    "metadata",
                                    {},
                                )
                            ),
                        )
                    )

            if playlist.items:

                playlist.current_index = min(
                    max(
                        -1,
                        playlist.current_index,
                    ),
                    len(playlist.items) - 1,
                )

            else:
                playlist.current_index = -1

            self.playlists[
                playlist_id
            ] = playlist

        active = data.get(
            "active_playlist_id"
        )

        if active in self.playlists:
            self.active_playlist_id = active
        else:
            self.active_playlist_id = None


__all__ = [
    "PlaylistItem",
    "Playlist",
    "PlaylistManager",
]


