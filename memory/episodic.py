"""
RENIX Episodic Memory
=====================

Stores and manages experiences/events that happened during RENIX sessions.

Episodic memory is different from ordinary long-term memory:

    Long-term memory:
        "The user prefers night study."

    Episodic memory:
        "On August 12, RENIX helped the user build the memory system."

An episode represents something that happened at a particular time and can
contain:

- what happened
- when it happened
- where it happened
- participants
- actions
- outcome
- emotional/contextual information
- related memories
- project/task information
- importance
- tags
- metadata

This module delegates persistence to MemoryStore.
Semantic/vector retrieval remains the responsibility of the dedicated
retrieval/vector-memory modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Iterable, List, Optional, Sequence
import logging
import uuid

from .memory_manager import (
    Memory,
    MemoryImportance,
    MemoryStatus,
    MemoryType,
)
from .memory_store import (
    MemoryNotFoundError,
    MemoryStore,
)


logger = logging.getLogger(__name__)


class EpisodicMemoryError(Exception):
    """Base exception for episodic-memory failures."""


@dataclass
class Episode:
    """
    Represents a single RENIX experience/event.

    Parameters
    ----------
    episode_id:
        Unique episode identifier.

    title:
        Human-readable episode title.

    summary:
        Concise description of what happened.

    started_at:
        Episode start timestamp.

    ended_at:
        Optional episode completion timestamp.

    location:
        Optional physical/logical location.

    participants:
        People/devices/agents involved.

    actions:
        Actions performed during the episode.

    outcome:
        Result of the episode.

    tags:
        Search/filter tags.

    metadata:
        Additional structured information.

    memory_ids:
        IDs of related Memory objects.

    importance:
        Episode importance.

    user_id:
        Associated RENIX user.

    session_id:
        Associated session.

    conversation_id:
        Associated conversation.

    project_id:
        Associated project.

    task_id:
        Associated task.

    emotional_context:
        Optional emotional/context state.

    source:
        Origin of the episode.

    status:
        Current episode state.
    """

    episode_id: str
    title: str
    summary: str

    started_at: str
    ended_at: Optional[str] = None

    location: Optional[str] = None

    participants: List[str] = field(
        default_factory=list
    )

    actions: List[str] = field(
        default_factory=list
    )

    outcome: Optional[str] = None

    tags: List[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    memory_ids: List[str] = field(
        default_factory=list
    )

    importance: MemoryImportance = (
        MemoryImportance.NORMAL
    )

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    project_id: Optional[str] = None
    task_id: Optional[str] = None

    emotional_context: Optional[str] = None

    source: Optional[str] = None

    status: MemoryStatus = (
        MemoryStatus.ACTIVE
    )

    created_at: str = field(
        default_factory=lambda: (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )

    updated_at: str = field(
        default_factory=lambda: (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )


class EpisodicMemory:
    """
    RENIX episodic-memory manager.

    Episodes can be used by the orchestrator to answer questions such as:

        "What did we do yesterday?"

        "What happened during the last coding session?"

        "What was the result of that task?"

        "When did I last work on this project?"

        "What happened before this decision?"

    Persistence is provided by MemoryStore for the underlying Memory records.
    """

    def __init__(
        self,
        store: MemoryStore,
    ) -> None:

        if store is None:
            raise ValueError(
                "EpisodicMemory requires a MemoryStore."
            )

        self.store = store

        self._episodes: dict[
            str,
            Episode,
        ] = {}

        self._lock = RLock()

    # ======================================================================
    # CREATE EPISODE
    # ======================================================================

    def create(
        self,
        title: str,
        summary: str,
        *,
        started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
        location: Optional[str] = None,
        participants: Optional[
            Iterable[str]
        ] = None,
        actions: Optional[
            Iterable[str]
        ] = None,
        outcome: Optional[str] = None,
        tags: Optional[
            Iterable[str]
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
        importance: MemoryImportance = (
            MemoryImportance.NORMAL
        ),
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        emotional_context: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Episode:
        """
        Create a new episode.
        """

        now = self._utc_now()

        episode = Episode(
            episode_id=self._new_id(),
            title=str(title),
            summary=str(summary),
            started_at=(
                started_at
                or now
            ),
            ended_at=ended_at,
            location=location,
            participants=self._clean_list(
                participants
            ),
            actions=self._clean_list(
                actions
            ),
            outcome=outcome,
            tags=self._clean_list(
                tags
            ),
            metadata=dict(
                metadata or {}
            ),
            importance=importance,
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            project_id=project_id,
            task_id=task_id,
            emotional_context=(
                emotional_context
            ),
            source=source,
            created_at=now,
            updated_at=now,
        )

        with self._lock:

            self._episodes[
                episode.episode_id
            ] = episode

        return episode

    # ======================================================================
    # RECORD EVENT
    # ======================================================================

    def record(
        self,
        title: str,
        summary: str,
        **kwargs: Any,
    ) -> Episode:
        """
        Alias for create().

        Useful when RENIX records an event directly from the orchestrator.
        """

        return self.create(
            title,
            summary,
            **kwargs,
        )

    # ======================================================================
    # COMPLETE EPISODE
    # ======================================================================

    def complete(
        self,
        episode_id: str,
        *,
        outcome: Optional[str] = None,
        ended_at: Optional[str] = None,
        additional_actions: Optional[
            Iterable[str]
        ] = None,
        importance: Optional[
            MemoryImportance
        ] = None,
    ) -> Optional[Episode]:
        """
        Complete an active episode.
        """

        with self._lock:

            episode = self._episodes.get(
                episode_id
            )

            if episode is None:
                return None

            episode.ended_at = (
                ended_at
                or self._utc_now()
            )

            if outcome is not None:
                episode.outcome = (
                    outcome
                )

            if additional_actions:
                episode.actions.extend(
                    self._clean_list(
                        additional_actions
                    )
                )

            if importance is not None:
                episode.importance = (
                    importance
                )

            episode.updated_at = (
                self._utc_now()
            )

            return episode

    # ======================================================================
    # READ
    # ======================================================================

    def get(
        self,
        episode_id: str,
    ) -> Optional[Episode]:
        """
        Retrieve an episode by ID.
        """

        with self._lock:

            return self._episodes.get(
                episode_id
            )

    def require(
        self,
        episode_id: str,
    ) -> Episode:
        """
        Retrieve an episode or raise an error.
        """

        episode = self.get(
            episode_id
        )

        if episode is None:
            raise EpisodicMemoryError(
                f"Episode not found: "
                f"{episode_id}"
            )

        return episode

    def exists(
        self,
        episode_id: str,
    ) -> bool:

        with self._lock:

            return (
                episode_id
                in self._episodes
            )

    # ======================================================================
    # UPDATE
    # ======================================================================

    def update(
        self,
        episode: Episode,
    ) -> Episode:
        """
        Replace/update an existing episode.
        """

        if not isinstance(
            episode,
            Episode,
        ):
            raise TypeError(
                "Expected an Episode object."
            )

        with self._lock:

            if (
                episode.episode_id
                not in self._episodes
            ):
                raise EpisodicMemoryError(
                    f"Episode does not exist: "
                    f"{episode.episode_id}"
                )

            episode.updated_at = (
                self._utc_now()
            )

            self._episodes[
                episode.episode_id
            ] = episode

            return episode

    def add_action(
        self,
        episode_id: str,
        action: str,
    ) -> Optional[Episode]:
        """
        Add an action to an episode.
        """

        with self._lock:

            episode = self.get(
                episode_id
            )

            if episode is None:
                return None

            action = str(
                action
            ).strip()

            if action:
                episode.actions.append(
                    action
                )

            episode.updated_at = (
                self._utc_now()
            )

            return episode

    def add_participant(
        self,
        episode_id: str,
        participant: str,
    ) -> Optional[Episode]:
        """
        Add a participant to an episode.
        """

        with self._lock:

            episode = self.get(
                episode_id
            )

            if episode is None:
                return None

            participant = str(
                participant
            ).strip()

            if (
                participant
                and participant
                not in episode.participants
            ):
                episode.participants.append(
                    participant
                )

            episode.updated_at = (
                self._utc_now()
            )

            return episode

    def add_tag(
        self,
        episode_id: str,
        tag: str,
    ) -> Optional[Episode]:
        """
        Add a tag to an episode.
        """

        with self._lock:

            episode = self.get(
                episode_id
            )

            if episode is None:
                return None

            tag = str(
                tag
            ).strip()

            if (
                tag
                and tag not in episode.tags
            ):
                episode.tags.append(
                    tag
                )

            episode.updated_at = (
                self._utc_now()
            )

            return episode

    # ======================================================================
    # LINK MEMORY
    # ======================================================================

    def attach_memory(
        self,
        episode_id: str,
        memory: Memory | str,
    ) -> bool:
        """
        Attach a Memory object or memory ID to an episode.

        If a Memory object is supplied, it is persisted through MemoryStore.
        """

        with self._lock:

            episode = self.get(
                episode_id
            )

            if episode is None:
                return False

            if isinstance(
                memory,
                Memory,
            ):

                if self.store.exists(
                    memory.memory_id
                ):
                    self.store.update(
                        memory
                    )
                else:
                    self.store.add(
                        memory
                    )

                memory_id = (
                    memory.memory_id
                )

            else:

                memory_id = str(
                    memory
                )

                if not self.store.exists(
                    memory_id
                ):
                    raise MemoryNotFoundError(
                        f"Memory not found: "
                        f"{memory_id}"
                    )

            if (
                memory_id
                not in episode.memory_ids
            ):
                episode.memory_ids.append(
                    memory_id
                )

            episode.updated_at = (
                self._utc_now()
            )

            return True

    def detach_memory(
        self,
        episode_id: str,
        memory_id: str,
    ) -> bool:
        """
        Remove a memory link from an episode.
        """

        with self._lock:

            episode = self.get(
                episode_id
            )

            if episode is None:
                return False

            if (
                memory_id
                not in episode.memory_ids
            ):
                return False

            episode.memory_ids.remove(
                memory_id
            )

            episode.updated_at = (
                self._utc_now()
            )

            return True

    def get_memories(
        self,
        episode_id: str,
    ) -> List[Memory]:
        """
        Retrieve Memory objects attached to an episode.
        """

        episode = self.get(
            episode_id
        )

        if episode is None:
            return []

        result: list[
            Memory
        ] = []

        for memory_id in (
            episode.memory_ids
        ):

            memory = self.store.get(
                memory_id
            )

            if memory is not None:
                result.append(
                    memory
                )

        return result

    # ======================================================================
    # SEARCH
    # ======================================================================

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> List[Episode]:
        """
        Search episodes using lexical matching.

        This intentionally stays lightweight. Semantic retrieval is handled
        elsewhere.
        """

        query = str(
            query
        ).strip()

        if not query:
            return []

        tokens = self._tokenize(
            query
        )

        if not tokens:
            return []

        candidates: list[
            tuple[
                float,
                Episode,
            ]
        ] = []

        with self._lock:

            episodes = list(
                self._episodes.values()
            )

        for episode in episodes:

            if (
                episode.status
                != MemoryStatus.ACTIVE
            ):
                continue

            if (
                user_id is not None
                and episode.user_id
                != user_id
            ):
                continue

            if (
                project_id is not None
                and episode.project_id
                != project_id
            ):
                continue

            if (
                task_id is not None
                and episode.task_id
                != task_id
            ):
                continue

            searchable = " ".join(
                [
                    episode.title,
                    episode.summary,
                    episode.outcome or "",
                    episode.location or "",
                    episode.emotional_context or "",
                    " ".join(
                        episode.actions
                    ),
                    " ".join(
                        episode.participants
                    ),
                    " ".join(
                        episode.tags
                    ),
                ]
            )

            episode_tokens = (
                self._tokenize(
                    searchable
                )
            )

            overlap = (
                tokens
                & episode_tokens
            )

            if not overlap:
                continue

            score = (
                len(overlap)
                / max(
                    len(tokens),
                    1,
                )
            )

            if (
                query.lower()
                in searchable.lower()
            ):
                score += 1.0

            candidates.append(
                (
                    score,
                    episode,
                )
            )

        candidates.sort(
            key=lambda item: (
                item[0],
                self._timestamp(
                    item[1].updated_at
                ),
            ),
            reverse=True,
        )

        return [
            episode
            for _, episode
            in candidates[:max(1, limit)]
        ]

    # ======================================================================
    # RECENT EPISODES
    # ======================================================================

    def recent(
        self,
        *,
        limit: int = 20,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> List[Episode]:
        """
        Return recently updated episodes.
        """

        with self._lock:

            episodes = list(
                self._episodes.values()
            )

        filtered = []

        for episode in episodes:

            if (
                episode.status
                != MemoryStatus.ACTIVE
            ):
                continue

            if (
                user_id is not None
                and episode.user_id
                != user_id
            ):
                continue

            if (
                session_id is not None
                and episode.session_id
                != session_id
            ):
                continue

            if (
                project_id is not None
                and episode.project_id
                != project_id
            ):
                continue

            if (
                task_id is not None
                and episode.task_id
                != task_id
            ):
                continue

            filtered.append(
                episode
            )

        filtered.sort(
            key=lambda episode: (
                self._timestamp(
                    episode.started_at
                )
            ),
            reverse=True,
        )

        return filtered[
            :max(1, int(limit))
        ]

    # ======================================================================
    # SESSION HISTORY
    # ======================================================================

    def for_session(
        self,
        session_id: str,
        *,
        limit: int = 100,
    ) -> List[Episode]:
        """
        Return all active episodes from a session.
        """

        return self.recent(
            limit=limit,
            session_id=session_id,
        )

    # ======================================================================
    # PROJECT HISTORY
    # ======================================================================

    def for_project(
        self,
        project_id: str,
        *,
        limit: int = 100,
    ) -> List[Episode]:
        """
        Return all active episodes related to a project.
        """

        return self.recent(
            limit=limit,
            project_id=project_id,
        )

    # ======================================================================
    # TASK HISTORY
    # ======================================================================

    def for_task(
        self,
        task_id: str,
        *,
        limit: int = 100,
    ) -> List[Episode]:
        """
        Return all active episodes related to a task.
        """

        return self.recent(
            limit=limit,
            task_id=task_id,
        )

    # ======================================================================
    # USER HISTORY
    # ======================================================================

    def for_user(
        self,
        user_id: str,
        *,
        limit: int = 100,
    ) -> List[Episode]:
        """
        Return all active episodes belonging to a user.
        """

        return self.recent(
            limit=limit,
            user_id=user_id,
        )

    # ======================================================================
    # TIMELINE
    # ======================================================================

    def timeline(
        self,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Episode]:
        """
        Return episodes occurring within a time range.

        Timestamps should be ISO-8601 strings.
        """

        start_ts = (
            self._timestamp(start)
            if start
            else None
        )

        end_ts = (
            self._timestamp(end)
            if end
            else None
        )

        with self._lock:

            episodes = list(
                self._episodes.values()
            )

        result = []

        for episode in episodes:

            if (
                episode.status
                != MemoryStatus.ACTIVE
            ):
                continue

            if (
                user_id is not None
                and episode.user_id
                != user_id
            ):
                continue

            if (
                project_id is not None
                and episode.project_id
                != project_id
            ):
                continue

            if (
                task_id is not None
                and episode.task_id
                != task_id
            ):
                continue

            episode_ts = (
                self._timestamp(
                    episode.started_at
                )
            )

            if (
                start_ts is not None
                and episode_ts < start_ts
            ):
                continue

            if (
                end_ts is not None
                and episode_ts > end_ts
            ):
                continue

            result.append(
                episode
            )

        result.sort(
            key=lambda episode: (
                self._timestamp(
                    episode.started_at
                )
            )
        )

        return result[
            :max(1, int(limit))
        ]

    # ======================================================================
    # SUMMARIZATION
    # ======================================================================

    def summarize(
        self,
        *,
        limit: int = 20,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """
        Build a compact human-readable episode history.

        This does not use an LLM; it produces deterministic text.
        """

        episodes = self.recent(
            limit=limit,
            user_id=user_id,
            project_id=project_id,
            task_id=task_id,
        )

        if not episodes:
            return (
                "No episodic memories found."
            )

        lines: list[
            str
        ] = []

        for episode in reversed(
            episodes
        ):

            line = (
                f"{episode.started_at} — "
                f"{episode.title}: "
                f"{episode.summary}"
            )

            if episode.outcome:
                line += (
                    f" Outcome: "
                    f"{episode.outcome}"
                )

            lines.append(
                line
            )

        return "\n".join(
            lines
        )

    # ======================================================================
    # ARCHIVE / DELETE
    # ======================================================================

    def archive(
        self,
        episode_id: str,
    ) -> bool:
        """
        Archive an episode.
        """

        with self._lock:

            episode = self._episodes.get(
                episode_id
            )

            if episode is None:
                return False

            episode.status = (
                MemoryStatus.ARCHIVED
            )

            episode.updated_at = (
                self._utc_now()
            )

            return True

    def restore(
        self,
        episode_id: str,
    ) -> bool:
        """
        Restore an archived episode.
        """

        with self._lock:

            episode = self._episodes.get(
                episode_id
            )

            if episode is None:
                return False

            episode.status = (
                MemoryStatus.ACTIVE
            )

            episode.updated_at = (
                self._utc_now()
            )

            return True

    def delete(
        self,
        episode_id: str,
        *,
        permanent: bool = False,
    ) -> bool:
        """
        Delete an episode.

        By default the episode is soft-deleted.
        """

        with self._lock:

            episode = self._episodes.get(
                episode_id
            )

            if episode is None:
                return False

            if permanent:

                del self._episodes[
                    episode_id
                ]

                return True

            episode.status = (
                MemoryStatus.DELETED
            )

            episode.updated_at = (
                self._utc_now()
            )

            return True

    # ======================================================================
    # EXPORT
    # ======================================================================

    def to_dict(
        self,
        episode_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Convert an episode into a JSON-compatible dictionary.
        """

        episode = self.get(
            episode_id
        )

        if episode is None:
            return None

        return {
            "episode_id": episode.episode_id,
            "title": episode.title,
            "summary": episode.summary,
            "started_at": episode.started_at,
            "ended_at": episode.ended_at,
            "location": episode.location,
            "participants": list(
                episode.participants
            ),
            "actions": list(
                episode.actions
            ),
            "outcome": episode.outcome,
            "tags": list(
                episode.tags
            ),
            "metadata": dict(
                episode.metadata
            ),
            "memory_ids": list(
                episode.memory_ids
            ),
            "importance": (
                episode.importance.value
            ),
            "user_id": episode.user_id,
            "session_id": episode.session_id,
            "conversation_id": (
                episode.conversation_id
            ),
            "project_id": episode.project_id,
            "task_id": episode.task_id,
            "emotional_context": (
                episode.emotional_context
            ),
            "source": episode.source,
            "status": episode.status.value,
            "created_at": episode.created_at,
            "updated_at": episode.updated_at,
        }

    def export_all(self) -> List[
        dict[str, Any]
    ]:
        """
        Export all episodes as dictionaries.
        """

        with self._lock:

            ids = list(
                self._episodes.keys()
            )

        result = []

        for episode_id in ids:

            data = self.to_dict(
                episode_id
            )

            if data is not None:
                result.append(
                    data
                )

        return result

    # ======================================================================
    # STATISTICS
    # ======================================================================

    def statistics(self) -> dict[str, Any]:
        """
        Return episodic-memory statistics.
        """

        with self._lock:

            episodes = list(
                self._episodes.values()
            )

        active = sum(
            1
            for episode in episodes
            if episode.status
            == MemoryStatus.ACTIVE
        )

        archived = sum(
            1
            for episode in episodes
            if episode.status
            == MemoryStatus.ARCHIVED
        )

        deleted = sum(
            1
            for episode in episodes
            if episode.status
            == MemoryStatus.DELETED
        )

        critical = sum(
            1
            for episode in episodes
            if episode.importance
            == MemoryImportance.CRITICAL
        )

        high = sum(
            1
            for episode in episodes
            if episode.importance
            == MemoryImportance.HIGH
        )

        linked_memories = sum(
            len(
                episode.memory_ids
            )
            for episode in episodes
        )

        return {
            "total": len(
                episodes
            ),
            "active": active,
            "archived": archived,
            "deleted": deleted,
            "critical": critical,
            "high": high,
            "linked_memories": (
                linked_memories
            ),
        }

    def health(self) -> dict[str, Any]:
        """
        Return health information.
        """

        stats = self.statistics()

        return {
            "status": "healthy",
            **stats,
        }

    # ======================================================================
    # INTERNAL HELPERS
    # ======================================================================

    @staticmethod
    def _new_id() -> str:
        return (
            "episode_"
            + uuid.uuid4().hex
        )

    @staticmethod
    def _utc_now() -> str:
        return (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

    @staticmethod
    def _clean_list(
        values: Optional[
            Iterable[str]
        ],
    ) -> List[str]:

        if values is None:
            return []

        result = []

        for value in values:

            value = str(
                value
            ).strip()

            if value:
                result.append(
                    value
                )

        return list(
            dict.fromkeys(
                result
            )
        )

    @staticmethod
    def _tokenize(
        text: str,
    ) -> set[str]:

        normalized = (
            str(text)
            .lower()
        )

        for char in (
            ",",
            ".",
            "!",
            "?",
            ":",
            ";",
            "/",
            "\\",
            "-",
            "_",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
        ):
            normalized = (
                normalized.replace(
                    char,
                    " ",
                )
            )

        return {
            token
            for token
            in normalized.split()
            if len(token) >= 2
        }

    @staticmethod
    def _timestamp(
        value: Optional[str],
    ) -> float:

        if not value:
            return 0.0

        try:

            return datetime.fromisoformat(
                str(value).replace(
                    "Z",
                    "+00:00",
                )
            ).timestamp()

        except (
            ValueError,
            TypeError,
        ):

            return 0.0


__all__ = [
    "Episode",
    "EpisodicMemory",
    "EpisodicMemoryError",
]


