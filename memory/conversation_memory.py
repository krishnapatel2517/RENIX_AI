"""
RENIX AI - Conversation Memory

Stores and manages conversational context for RENIX.

Responsibilities:
- Store conversation messages
- Maintain conversation history
- Track sessions and conversations
- Retrieve recent context
- Search conversation history
- Summarize conversations
- Mark important messages
- Preserve conversation metadata
- Support long-running conversations
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


logger = logging.getLogger(
    "RENIX.memory.conversation_memory"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ConversationMessage:
    """A single message in a RENIX conversation."""

    message_id: str
    conversation_id: str
    role: str
    content: str

    timestamp: str = field(
        default_factory=_now
    )

    session_id: Optional[str] = None

    user_id: Optional[str] = None

    message_type: str = "text"

    important: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "message_type": self.message_type,
            "important": self.important,
            "metadata": self.metadata,
        }


@dataclass
class Conversation:
    """Represents a complete RENIX conversation."""

    conversation_id: str

    title: str = "New Conversation"

    created_at: str = field(
        default_factory=_now
    )

    updated_at: str = field(
        default_factory=_now
    )

    session_id: Optional[str] = None

    user_id: Optional[str] = None

    messages: List[
        ConversationMessage
    ] = field(default_factory=list)

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    archived: bool = False

    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "messages": [
                message.to_dict()
                for message in self.messages
            ],
            "metadata": self.metadata,
            "archived": self.archived,
            "summary": self.summary,
        }


class ConversationMemory:
    """
    Conversation memory subsystem for RENIX.

    This class intentionally keeps an in-memory cache and can optionally
    connect to a persistent storage backend.
    """

    def __init__(
        self,
        store: Optional[Any] = None,
        max_messages_per_conversation: int = 1000,
    ) -> None:

        self.store = store

        self.max_messages = max(
            1,
            int(
                max_messages_per_conversation
            ),
        )

        self._conversations: Dict[
            str,
            Conversation,
        ] = {}

        self._lock = threading.RLock()

        logger.info(
            "ConversationMemory initialized."
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> bool:

        if self.store is not None:

            initialize = getattr(
                self.store,
                "initialize",
                None,
            )

            if callable(initialize):

                try:
                    initialize()

                except Exception:
                    logger.exception(
                        "Conversation store initialization failed."
                    )

                    return False

        return True

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def create_conversation(
        self,
        title: str = "New Conversation",
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> str:

        conversation_id = str(
            uuid.uuid4()
        )

        conversation = Conversation(
            conversation_id=conversation_id,
            title=title.strip()
            or "New Conversation",
            session_id=session_id,
            user_id=user_id,
            metadata=dict(
                metadata or {}
            ),
        )

        with self._lock:

            self._conversations[
                conversation_id
            ] = conversation

        self._persist(
            conversation
        )

        return conversation_id

    def get_conversation(
        self,
        conversation_id: str,
    ) -> Optional[Conversation]:

        with self._lock:

            conversation = (
                self._conversations.get(
                    conversation_id
                )
            )

        if conversation is not None:
            return conversation

        if self.store is not None:

            method = getattr(
                self.store,
                "get_conversation",
                None,
            )

            if callable(method):

                try:

                    conversation = method(
                        conversation_id
                    )

                    if isinstance(
                        conversation,
                        Conversation,
                    ):
                        with self._lock:
                            self._conversations[
                                conversation_id
                            ] = conversation

                        return conversation

                except Exception:
                    logger.exception(
                        "Failed loading conversation."
                    )

        return None

    def delete_conversation(
        self,
        conversation_id: str,
    ) -> bool:

        with self._lock:

            conversation = (
                self._conversations.pop(
                    conversation_id,
                    None,
                )
            )

        if conversation is None:
            return False

        if self.store is not None:

            method = getattr(
                self.store,
                "delete_conversation",
                None,
            )

            if callable(method):

                try:
                    method(
                        conversation_id
                    )
                except Exception:
                    logger.exception(
                        "Failed deleting persistent conversation."
                    )

        return True

    def archive_conversation(
        self,
        conversation_id: str,
    ) -> bool:

        conversation = self.get_conversation(
            conversation_id
        )

        if conversation is None:
            return False

        conversation.archived = True

        conversation.updated_at = _now()

        self._persist(
            conversation
        )

        return True

    def restore_conversation(
        self,
        conversation_id: str,
    ) -> bool:

        conversation = self.get_conversation(
            conversation_id
        )

        if conversation is None:
            return False

        conversation.archived = False

        conversation.updated_at = _now()

        self._persist(
            conversation
        )

        return True

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        message_type: str = "text",
        important: bool = False,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Optional[str]:

        conversation = self.get_conversation(
            conversation_id
        )

        if conversation is None:
            logger.warning(
                "Conversation does not exist: %s",
                conversation_id,
            )
            return None

        content = str(
            content
        ).strip()

        if not content:
            return None

        message = ConversationMessage(
            message_id=str(
                uuid.uuid4()
            ),
            conversation_id=conversation_id,
            role=str(role),
            content=content,
            session_id=(
                session_id
                or conversation.session_id
            ),
            user_id=(
                user_id
                or conversation.user_id
            ),
            message_type=message_type,
            important=important,
            metadata=dict(
                metadata or {}
            ),
        )

        with self._lock:

            conversation.messages.append(
                message
            )

            if (
                len(
                    conversation.messages
                )
                > self.max_messages
            ):

                conversation.messages = (
                    self._trim_messages(
                        conversation.messages
                    )
                )

            conversation.updated_at = _now()

        self._persist(
            conversation
        )

        return message.message_id

    def add_user_message(
        self,
        conversation_id: str,
        content: str,
        **kwargs: Any,
    ) -> Optional[str]:

        return self.add_message(
            conversation_id,
            "user",
            content,
            **kwargs,
        )

    def add_assistant_message(
        self,
        conversation_id: str,
        content: str,
        **kwargs: Any,
    ) -> Optional[str]:

        return self.add_message(
            conversation_id,
            "assistant",
            content,
            **kwargs,
        )

    def add_system_message(
        self,
        conversation_id: str,
        content: str,
        **kwargs: Any,
    ) -> Optional[str]:

        return self.add_message(
            conversation_id,
            "system",
            content,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(
        self,
        conversation_id: str,
        *,
        limit: Optional[int] = None,
        include_system: bool = True,
    ) -> List[
        ConversationMessage
    ]:

        conversation = self.get_conversation(
            conversation_id
        )

        if conversation is None:
            return []

        messages = list(
            conversation.messages
        )

        if not include_system:

            messages = [
                message
                for message in messages
                if message.role
                != "system"
            ]

        if limit is not None:

            limit = max(
                1,
                int(limit),
            )

            messages = messages[
                -limit:
            ]

        return messages

    def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 20,
    ) -> List[
        ConversationMessage
    ]:

        return self.get_history(
            conversation_id,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # LLM-ready context
    # ------------------------------------------------------------------

    def get_context(
        self,
        conversation_id: str,
        *,
        limit: int = 20,
    ) -> List[
        Dict[str, str]
    ]:

        messages = self.get_history(
            conversation_id,
            limit=limit,
        )

        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        conversation_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[
        ConversationMessage
    ]:

        query = str(
            query
        ).strip().lower()

        if not query:
            return []

        candidates: List[
            ConversationMessage
        ] = []

        if conversation_id:

            conversation = (
                self.get_conversation(
                    conversation_id
                )
            )

            if conversation:
                candidates.extend(
                    conversation.messages
                )

        else:

            with self._lock:

                for conversation in (
                    self._conversations.values()
                ):
                    candidates.extend(
                        conversation.messages
                    )

        scored = []

        for message in candidates:

            content = (
                message.content.lower()
            )

            if query in content:

                score = 1.0

            else:

                words = set(
                    query.split()
                )

                content_words = set(
                    content.split()
                )

                overlap = (
                    words
                    & content_words
                )

                if not overlap:
                    continue

                score = (
                    len(overlap)
                    / max(
                        1,
                        len(words),
                    )
                )

            if message.important:
                score += 0.1

            scored.append(
                (
                    score,
                    message,
                )
            )

        scored.sort(
            key=lambda item: (
                item[0],
                item[1].timestamp,
            ),
            reverse=True,
        )

        return [
            message
            for _, message in scored[
                :max(
                    1,
                    int(limit),
                ):
            ]
        ]

    # ------------------------------------------------------------------
    # Important memories
    # ------------------------------------------------------------------

    def mark_important(
        self,
        conversation_id: str,
        message_id: str,
        important: bool = True,
    ) -> bool:

        conversation = self.get_conversation(
            conversation_id
        )

        if conversation is None:
            return False

        for message in conversation.messages:

            if (
                message.message_id
                == message_id
            ):

                message.important = (
                    important
                )

                conversation.updated_at = (
                    _now()
                )

                self._persist(
                    conversation
                )

                return True

        return False

    def get_important_messages(
        self,
        conversation_id: str,
    ) -> List[
        ConversationMessage
    ]:

        conversation = self.get_conversation(
            conversation_id
        )

        if conversation is None:
            return []

        return [
            message
            for message in conversation.messages
            if message.important
        ]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def set_summary(
        self,
        conversation_id: str,
        summary: str,
    ) -> bool:

        conversation = self.get_conversation(
            conversation_id
        )

        if conversation is None:
            return False

        conversation.summary = str(
            summary
        ).strip()

        conversation.updated_at = _now()

        self._persist(
            conversation
        )

        return True

    def get_summary(
        self,
        conversation_id: str,
    ) -> Optional[str]:

        conversation = self.get_conversation(
            conversation_id
        )

        if conversation is None:
            return None

        return conversation.summary

    # ------------------------------------------------------------------
    # Conversation listing
    # ------------------------------------------------------------------

    def list_conversations(
        self,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        include_archived: bool = False,
    ) -> List[Conversation]:

        with self._lock:

            conversations = list(
                self._conversations.values()
            )

        filtered = []

        for conversation in conversations:

            if (
                conversation.archived
                and not include_archived
            ):
                continue

            if (
                user_id is not None
                and conversation.user_id
                != user_id
            ):
                continue

            if (
                session_id is not None
                and conversation.session_id
                != session_id
            ):
                continue

            filtered.append(
                conversation
            )

        filtered.sort(
            key=lambda item: (
                item.updated_at
            ),
            reverse=True,
        )

        return filtered

    # ------------------------------------------------------------------
    # Persistence adapter
    # ------------------------------------------------------------------

    def _persist(
        self,
        conversation: Conversation,
    ) -> None:

        if self.store is None:
            return

        for method_name in (
            "save_conversation",
            "save",
            "store",
            "upsert",
        ):

            method = getattr(
                self.store,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:

                method(
                    conversation.to_dict()
                )

                return

            except TypeError:

                try:

                    method(
                        conversation
                    )

                    return

                except Exception:
                    continue

            except Exception:
                logger.exception(
                    "Conversation persistence failed."
                )
                return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_messages(
        messages: List[
            ConversationMessage
        ],
    ) -> List[
        ConversationMessage
    ]:

        if not messages:
            return []

        important = [
            message
            for message in messages
            if message.important
        ]

        recent = messages[-900:]

        existing = {
            message.message_id
            for message in recent
        }

        for message in important:

            if message.message_id not in existing:

                recent.insert(
                    0,
                    message,
                )

        return recent[-1000:]


__all__ = [
    "ConversationMessage",
    "Conversation",
    "ConversationMemory",
]


