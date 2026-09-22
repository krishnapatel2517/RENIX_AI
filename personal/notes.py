"""
RENIX Personal Notes

Manages personal notes with titles, content, categories,
tags, search, pinning, archiving, and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable
import uuid


@dataclass
class Note:
    """Represents a RENIX personal note."""

    title: str
    content: str = ""
    category: str = ""
    tags: list[str] = field(default_factory=list)
    note_id: str | None = None
    created_at: datetime = field(
        default_factory=datetime.now
    )
    updated_at: datetime = field(
        default_factory=datetime.now
    )
    pinned: bool = False
    archived: bool = False
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        if not self.note_id:
            self.note_id = (
                f"NOTE-{uuid.uuid4().hex[:12].upper()}"
            )

        self.title = (
            self.title or ""
        ).strip()

        if not self.title:
            raise ValueError(
                "Note title cannot be empty."
            )

        if not isinstance(
            self.created_at,
            datetime,
        ):
            self.created_at = datetime.now()

        if not isinstance(
            self.updated_at,
            datetime,
        ):
            self.updated_at = self.created_at

    def update_content(
        self,
        content: str,
    ) -> None:

        self.content = content or ""
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:

        return {
            "note_id": self.note_id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "tags": list(self.tags),
            "created_at": (
                self.created_at.isoformat()
            ),
            "updated_at": (
                self.updated_at.isoformat()
            ),
            "pinned": self.pinned,
            "archived": self.archived,
            "metadata": dict(self.metadata),
        }


class NotesManager:
    """Central personal-notes manager for RENIX."""

    def __init__(self) -> None:

        self.notes: dict[
            str,
            Note,
        ] = {}

    # ============================================================
    # CREATE
    # ============================================================

    def create_note(
        self,
        title: str,
        content: str = "",
        *,
        category: str = "",
        tags: Iterable[str] | None = None,
        note_id: str | None = None,
        pinned: bool = False,
        archived: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Note:

        note = Note(
            title=title,
            content=content,
            category=category,
            tags=list(tags or []),
            note_id=note_id,
            pinned=pinned,
            archived=archived,
            metadata=dict(
                metadata or {}
            ),
        )

        if note.note_id in self.notes:
            raise ValueError(
                f"Note '{note.note_id}' already exists."
            )

        self.notes[
            note.note_id
        ] = note

        return note

    def add_note(
        self,
        note: Note,
    ) -> str:

        if not isinstance(
            note,
            Note,
        ):
            raise TypeError(
                "note must be a Note."
            )

        if note.note_id in self.notes:
            raise ValueError(
                f"Note '{note.note_id}' already exists."
            )

        self.notes[
            note.note_id
        ] = note

        return note.note_id

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def get_note(
        self,
        note_id: str,
    ) -> Note | None:

        return self.notes.get(
            note_id
        )

    def get_all_notes(
        self,
        *,
        include_archived: bool = False,
    ) -> list[Note]:

        notes = list(
            self.notes.values()
        )

        if not include_archived:
            notes = [
                note
                for note in notes
                if not note.archived
            ]

        return sorted(
            notes,
            key=self._sort_key,
        )

    def get_pinned_notes(self) -> list[Note]:

        return [
            note
            for note in self.get_all_notes()
            if note.pinned
        ]

    def get_archived_notes(self) -> list[Note]:

        return [
            note
            for note in self.notes.values()
            if note.archived
        ]

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        *,
        include_archived: bool = False,
    ) -> list[Note]:

        query = (
            query or ""
        ).strip().casefold()

        if not query:
            return []

        results = []

        for note in self.notes.values():

            if (
                note.archived
                and not include_archived
            ):
                continue

            searchable = " ".join(
                [
                    note.title,
                    note.content,
                    note.category,
                    " ".join(note.tags),
                ]
            ).casefold()

            if query in searchable:
                results.append(note)

        return sorted(
            results,
            key=self._sort_key,
        )

    def search_title(
        self,
        title: str,
    ) -> list[Note]:

        title = (
            title or ""
        ).strip().casefold()

        if not title:
            return []

        return [
            note
            for note in self.notes.values()
            if title in note.title.casefold()
        ]

    def by_category(
        self,
        category: str,
    ) -> list[Note]:

        category = (
            category or ""
        ).strip().casefold()

        return sorted(
            [
                note
                for note in self.notes.values()
                if note.category.casefold()
                == category
            ],
            key=self._sort_key,
        )

    def by_tag(
        self,
        tag: str,
    ) -> list[Note]:

        tag = (
            tag or ""
        ).strip().casefold()

        return sorted(
            [
                note
                for note in self.notes.values()
                if any(
                    tag == item.casefold()
                    for item in note.tags
                )
            ],
            key=self._sort_key,
        )

    # ============================================================
    # UPDATE
    # ============================================================

    def update_note(
        self,
        note_id: str,
        **changes: Any,
    ) -> bool:

        note = self.get_note(
            note_id
        )

        if note is None:
            return False

        allowed_fields = {
            "title",
            "content",
            "category",
            "tags",
            "pinned",
            "archived",
            "metadata",
        }

        changed = False

        for key, value in changes.items():

            if key not in allowed_fields:
                continue

            if key == "title":

                value = str(
                    value
                ).strip()

                if not value:
                    raise ValueError(
                        "Note title cannot be empty."
                    )

            setattr(
                note,
                key,
                value,
            )

            changed = True

        if changed:
            note.updated_at = datetime.now()

        return True

    def append_content(
        self,
        note_id: str,
        content: str,
    ) -> bool:

        note = self.get_note(
            note_id
        )

        if note is None:
            return False

        content = content or ""

        if note.content:
            note.content += "\n" + content
        else:
            note.content = content

        note.updated_at = datetime.now()

        return True

    # ============================================================
    # PINNING
    # ============================================================

    def pin(
        self,
        note_id: str,
    ) -> bool:

        note = self.get_note(
            note_id
        )

        if note is None:
            return False

        note.pinned = True
        note.updated_at = datetime.now()

        return True

    def unpin(
        self,
        note_id: str,
    ) -> bool:

        note = self.get_note(
            note_id
        )

        if note is None:
            return False

        note.pinned = False
        note.updated_at = datetime.now()

        return True

    def toggle_pin(
        self,
        note_id: str,
    ) -> bool:

        note = self.get_note(
            note_id
        )

        if note is None:
            return False

        note.pinned = not note.pinned
        note.updated_at = datetime.now()

        return note.pinned

    # ============================================================
    # ARCHIVING
    # ============================================================

    def archive(
        self,
        note_id: str,
    ) -> bool:

        note = self.get_note(
            note_id
        )

        if note is None:
            return False

        note.archived = True
        note.updated_at = datetime.now()

        return True

    def restore(
        self,
        note_id: str,
    ) -> bool:

        note = self.get_note(
            note_id
        )

        if note is None:
            return False

        note.archived = False
        note.updated_at = datetime.now()

        return True

    # ============================================================
    # TAG MANAGEMENT
    # ============================================================

    def add_tag(
        self,
        note_id: str,
        tag: str,
    ) -> bool:

        note = self.get_note(
            note_id
        )

        if note is None:
            return False

        tag = (
            tag or ""
        ).strip()

        if not tag:
            return False

        if tag.casefold() not in {
            item.casefold()
            for item in note.tags
        }:
            note.tags.append(tag)
            note.updated_at = datetime.now()

        return True

    def remove_tag(
        self,
        note_id: str,
        tag: str,
    ) -> bool:

        note = self.get_note(
            note_id
        )

        if note is None:
            return False

        tag = (
            tag or ""
        ).strip().casefold()

        for existing in list(
            note.tags
        ):
            if existing.casefold() == tag:
                note.tags.remove(existing)
                note.updated_at = datetime.now()
                return True

        return False

    # ============================================================
    # DELETE
    # ============================================================

    def delete_note(
        self,
        note_id: str,
    ) -> bool:

        if note_id not in self.notes:
            return False

        del self.notes[
            note_id
        ]

        return True

    def clear_archived(self) -> int:

        ids = [
            note_id
            for note_id, note
            in self.notes.items()
            if note.archived
        ]

        for note_id in ids:
            del self.notes[
                note_id
            ]

        return len(ids)

    # ============================================================
    # STATISTICS
    # ============================================================

    def statistics(self) -> dict[str, Any]:

        notes = list(
            self.notes.values()
        )

        return {
            "total": len(notes),
            "active": sum(
                not note.archived
                for note in notes
            ),
            "archived": sum(
                note.archived
                for note in notes
            ),
            "pinned": sum(
                note.pinned
                for note in notes
            ),
            "categories": len(
                {
                    note.category
                    for note in notes
                    if note.category
                }
            ),
            "tags": len(
                {
                    tag.casefold()
                    for note in notes
                    for tag in note.tags
                }
            ),
        }

    # ============================================================
    # SIMPLE COMMAND HANDLER
    # ============================================================

    def handle_command(
        self,
        command: str,
    ) -> dict[str, Any]:

        command = (
            command or ""
        ).strip()

        if not command:
            return {
                "success": False,
                "message": "Empty note command.",
            }

        lowered = command.casefold()

        if lowered in {
            "notes",
            "show notes",
            "show my notes",
        }:

            return {
                "success": True,
                "notes": [
                    note.to_dict()
                    for note in self.get_all_notes()
                ],
            }

        if lowered in {
            "pinned notes",
            "show pinned notes",
        }:

            return {
                "success": True,
                "notes": [
                    note.to_dict()
                    for note in self.get_pinned_notes()
                ],
            }

        if lowered.startswith(
            "find note "
        ):

            query = command[
                len("find note "):
            ].strip()

            return {
                "success": True,
                "notes": [
                    note.to_dict()
                    for note in self.search(query)
                ],
            }

        if lowered.startswith(
            "delete note "
        ):

            note_id = command[
                len("delete note "):
            ].strip()

            return {
                "success": self.delete_note(
                    note_id
                ),
                "note_id": note_id,
            }

        if lowered.startswith(
            "pin note "
        ):

            note_id = command[
                len("pin note "):
            ].strip()

            return {
                "success": self.pin(
                    note_id
                ),
                "note_id": note_id,
            }

        if lowered.startswith(
            "archive note "
        ):

            note_id = command[
                len("archive note "):
            ].strip()

            return {
                "success": self.archive(
                    note_id
                ),
                "note_id": note_id,
            }

        return {
            "success": False,
            "message": "Unknown note command.",
        }

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        return {
            "notes": [
                note.to_dict()
                for note in self.notes.values()
            ]
        }

    def import_state(
        self,
        data: dict[str, Any],
    ) -> None:

        if not isinstance(data, dict):
            raise TypeError(
                "Note state must be a dictionary."
            )

        self.notes.clear()

        raw_notes = data.get(
            "notes",
            [],
        )

        if not isinstance(
            raw_notes,
            list,
        ):
            return

        for raw in raw_notes:

            if not isinstance(
                raw,
                dict,
            ):
                continue

            try:

                created_raw = raw.get(
                    "created_at"
                )

                updated_raw = raw.get(
                    "updated_at"
                )

                created_at = (
                    datetime.fromisoformat(
                        created_raw
                    )
                    if created_raw
                    else datetime.now()
                )

                updated_at = (
                    datetime.fromisoformat(
                        updated_raw
                    )
                    if updated_raw
                    else created_at
                )

                note = Note(
                    title=str(
                        raw.get(
                            "title",
                            "Untitled Note",
                        )
                    ),
                    content=str(
                        raw.get(
                            "content",
                            "",
                        )
                    ),
                    category=str(
                        raw.get(
                            "category",
                            "",
                        )
                    ),
                    tags=list(
                        raw.get(
                            "tags",
                            [],
                        )
                    ),
                    note_id=raw.get(
                        "note_id"
                    ),
                    created_at=created_at,
                    updated_at=updated_at,
                    pinned=bool(
                        raw.get(
                            "pinned",
                            False,
                        )
                    ),
                    archived=bool(
                        raw.get(
                            "archived",
                            False,
                        )
                    ),
                    metadata=dict(
                        raw.get(
                            "metadata",
                            {},
                        )
                    ),
                )

                self.notes[
                    note.note_id
                ] = note

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

    # ============================================================
    # INTERNAL
    # ============================================================

    @staticmethod
    def _sort_key(
        note: Note,
    ) -> tuple:

        return (
            not note.pinned,
            -note.updated_at.timestamp(),
            note.title.casefold(),
        )


__all__ = [
    "Note",
    "NotesManager",
]


