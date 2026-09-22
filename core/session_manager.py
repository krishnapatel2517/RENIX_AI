"""
RENIX AI
Core Session Manager

Responsible for managing RENIX runtime sessions.

Features:
- Session creation
- Session IDs
- Session lifecycle
- Active session tracking
- Session metadata
- User/session context
- Session history
- Session variables
- Session timeout
- Session expiration
- Session switching
- Session export/import
- Session cleanup
- Session statistics
"""

from __future__ import annotations

import copy
import logging
import time
import uuid

from dataclasses import dataclass, field
from typing import Any, Optional


logger = logging.getLogger(
    "RENIX.SessionManager"
)


# ============================================================================
# SESSION
# ============================================================================

@dataclass
class Session:
    """
    Represents one RENIX conversation/runtime session.
    """

    session_id: str

    created_at: float = field(
        default_factory=time.time
    )

    last_activity: float = field(
        default_factory=time.time
    )

    user_id: Optional[str] = None

    name: Optional[str] = None

    active: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    variables: dict[str, Any] = field(
        default_factory=dict
    )

    context: dict[str, Any] = field(
        default_factory=dict
    )

    history: list[dict[str, Any]] = field(
        default_factory=list
    )

    max_history: int = 500

    timeout: Optional[float] = None

    def touch(self) -> None:
        """
        Update the last activity timestamp.
        """

        self.last_activity = time.time()

    def expired(self) -> bool:
        """
        Check whether the session has expired.
        """

        if not self.active:

            return True

        if self.timeout is None:

            return False

        return (
            time.time()
            - self.last_activity
            >= self.timeout
        )

    def add_history(
        self,
        entry: dict[str, Any],
    ) -> None:
        """
        Add an entry to session history.
        """

        self.history.append(
            copy.deepcopy(entry)
        )

        if len(
            self.history
        ) > self.max_history:

            self.history = self.history[
                -self.max_history:
            ]

        self.touch()

    def set_variable(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Store a session variable.
        """

        self.variables[
            key
        ] = copy.deepcopy(
            value
        )

        self.touch()

    def get_variable(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a session variable.
        """

        self.touch()

        return copy.deepcopy(
            self.variables.get(
                key,
                default,
            )
        )

    def delete_variable(
        self,
        key: str,
    ) -> bool:
        """
        Delete a session variable.
        """

        if key not in self.variables:

            return False

        del self.variables[
            key
        ]

        self.touch()

        return True

    def set_context(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Store session context.
        """

        self.context[
            key
        ] = copy.deepcopy(
            value
        )

        self.touch()

    def get_context(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve session context.
        """

        self.touch()

        return copy.deepcopy(
            self.context.get(
                key,
                default,
            )
        )

    def clear_context(
        self,
    ) -> None:
        """
        Clear session context.
        """

        self.context.clear()

        self.touch()

    def deactivate(
        self,
    ) -> None:
        """
        Deactivate the session.
        """

        self.active = False

        self.touch()

    def activate(
        self,
    ) -> None:
        """
        Activate the session.
        """

        self.active = True

        self.touch()

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert session into a serializable dictionary.
        """

        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "user_id": self.user_id,
            "name": self.name,
            "active": self.active,
            "metadata": copy.deepcopy(
                self.metadata
            ),
            "variables": copy.deepcopy(
                self.variables
            ),
            "context": copy.deepcopy(
                self.context
            ),
            "history": copy.deepcopy(
                self.history
            ),
            "max_history": self.max_history,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "Session":
        """
        Create a Session from a dictionary.
        """

        return cls(
            session_id=str(
                data[
                    "session_id"
                ]
            ),
            created_at=float(
                data.get(
                    "created_at",
                    time.time(),
                )
            ),
            last_activity=float(
                data.get(
                    "last_activity",
                    time.time(),
                )
            ),
            user_id=data.get(
                "user_id"
            ),
            name=data.get(
                "name"
            ),
            active=bool(
                data.get(
                    "active",
                    True,
                )
            ),
            metadata=copy.deepcopy(
                data.get(
                    "metadata",
                    {},
                )
            ),
            variables=copy.deepcopy(
                data.get(
                    "variables",
                    {},
                )
            ),
            context=copy.deepcopy(
                data.get(
                    "context",
                    {},
                )
            ),
            history=copy.deepcopy(
                data.get(
                    "history",
                    [],
                )
            ),
            max_history=int(
                data.get(
                    "max_history",
                    500,
                )
            ),
            timeout=data.get(
                "timeout"
            ),
        )


# ============================================================================
# SESSION MANAGER
# ============================================================================

class SessionManager:
    """
    Central session manager for RENIX.
    """

    def __init__(
        self,
        *,
        default_timeout: Optional[
            float
        ] = None,
        max_sessions: int = 100,
        default_history_limit: int = 500,
    ) -> None:

        self.logger = logger

        self.default_timeout = (
            default_timeout
        )

        self.max_sessions = max(
            1,
            int(max_sessions),
        )

        self.default_history_limit = max(
            1,
            int(
                default_history_limit
            ),
        )

        self.sessions: dict[
            str,
            Session,
        ] = {}

        self.active_session_id: Optional[
            str
        ] = None

        self.initialized = False

        self.started_at = time.time()

        self.total_created = 0

        self.total_closed = 0

        self.total_switched = 0

        self.total_messages = 0

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize the Session Manager.
        """

        if self.initialized:

            return

        self.initialized = True

        self.logger.info(
            "RENIX Session Manager initialized."
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Shut down the Session Manager.
        """

        self.initialized = False

        self.logger.info(
            "RENIX Session Manager shutdown."
        )

    # ========================================================================
    # CREATE SESSION
    # ========================================================================

    def create_session(
        self,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
        timeout: Optional[
            float
        ] = None,
        activate: bool = True,
    ) -> Session:
        """
        Create a new RENIX session.
        """

        self.cleanup_expired()

        if len(
            self.sessions
        ) >= self.max_sessions:

            self._remove_oldest_session()

        if session_id is None:

            session_id = (
                f"session_{uuid.uuid4().hex}"
            )

        if session_id in self.sessions:

            raise ValueError(
                f"Session already exists: "
                f"{session_id}"
            )

        session = Session(
            session_id=session_id,
            user_id=user_id,
            name=name,
            metadata=copy.deepcopy(
                metadata or {}
            ),
            timeout=(
                self.default_timeout
                if timeout is None
                else timeout
            ),
            max_history=(
                self.default_history_limit
            ),
        )

        self.sessions[
            session_id
        ] = session

        self.total_created += 1

        if activate:

            self.switch_session(
                session_id
            )

        self.logger.info(
            "Created session %s.",
            session_id,
        )

        return session

    # ========================================================================
    # GET SESSION
    # ========================================================================

    def get_session(
        self,
        session_id: Optional[str] = None,
    ) -> Optional[Session]:
        """
        Get a session.

        If no session ID is supplied,
        returns the active session.
        """

        self.cleanup_expired()

        if session_id is None:

            session_id = (
                self.active_session_id
            )

        if session_id is None:

            return None

        return self.sessions.get(
            session_id
        )

    # ========================================================================
    # REQUIRE SESSION
    # ========================================================================

    def require_session(
        self,
        session_id: Optional[str] = None,
    ) -> Session:
        """
        Get a session or raise an error.
        """

        session = self.get_session(
            session_id
        )

        if session is None:

            raise RuntimeError(
                "No active RENIX session."
            )

        return session

    # ========================================================================
    # ACTIVE SESSION
    # ========================================================================

    def get_active_session(
        self,
    ) -> Optional[Session]:
        """
        Return the currently active session.
        """

        return self.get_session()

    # ========================================================================
    # SWITCH SESSION
    # ========================================================================

    def switch_session(
        self,
        session_id: str,
    ) -> Session:
        """
        Switch the active RENIX session.
        """

        session = self.sessions.get(
            session_id
        )

        if session is None:

            raise KeyError(
                f"Unknown session: "
                f"{session_id}"
            )

        if session.expired():

            session.deactivate()

            raise RuntimeError(
                f"Session has expired: "
                f"{session_id}"
            )

        if (
            self.active_session_id
            != session_id
        ):

            self.total_switched += 1

        if (
            self.active_session_id
            is not None
        ):

            current = self.sessions.get(
                self.active_session_id
            )

            if current is not None:

                current.active = False

        session.activate()

        self.active_session_id = (
            session_id
        )

        self.logger.info(
            "Switched to session %s.",
            session_id,
        )

        return session

    # ========================================================================
    # CLOSE SESSION
    # ========================================================================

    def close_session(
        self,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Close and remove a session.
        """

        if session_id is None:

            session_id = (
                self.active_session_id
            )

        if session_id is None:

            return False

        session = self.sessions.get(
            session_id
        )

        if session is None:

            return False

        session.deactivate()

        del self.sessions[
            session_id
        ]

        self.total_closed += 1

        if (
            self.active_session_id
            == session_id
        ):

            self.active_session_id = None

            self._activate_latest_session()

        self.logger.info(
            "Closed session %s.",
            session_id,
        )

        return True

    # ========================================================================
    # END SESSION
    # ========================================================================

    def end_session(
        self,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Alias for close_session.
        """

        return self.close_session(
            session_id
        )

    # ========================================================================
    # LIST SESSIONS
    # ========================================================================

    def list_sessions(
        self,
        *,
        active_only: bool = False,
    ) -> list[Session]:
        """
        Return all sessions.
        """

        self.cleanup_expired()

        sessions = list(
            self.sessions.values()
        )

        if active_only:

            sessions = [
                session
                for session in sessions
                if session.active
            ]

        return sessions

    # ========================================================================
    # SESSION EXISTS
    # ========================================================================

    def exists(
        self,
        session_id: str,
    ) -> bool:
        """
        Check whether a session exists.
        """

        return (
            session_id
            in self.sessions
        )

    # ========================================================================
    # TOUCH SESSION
    # ========================================================================

    def touch(
        self,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Update session activity.
        """

        session = self.require_session(
            session_id
        )

        session.touch()

    # ========================================================================
    # ADD MESSAGE
    # ========================================================================

    def add_message(
        self,
        role: str,
        content: Any,
        *,
        session_id: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> dict[str, Any]:
        """
        Add a conversation message to a session.
        """

        session = self.require_session(
            session_id
        )

        message = {
            "id": (
                f"message_"
                f"{uuid.uuid4().hex}"
            ),
            "role": role,
            "content": copy.deepcopy(
                content
            ),
            "timestamp": time.time(),
            "metadata": copy.deepcopy(
                metadata or {}
            ),
        }

        session.add_history(
            message
        )

        self.total_messages += 1

        return copy.deepcopy(
            message
        )

    # ========================================================================
    # GET HISTORY
    # ========================================================================

    def get_history(
        self,
        *,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Return conversation history.
        """

        session = self.require_session(
            session_id
        )

        history = session.history

        if limit is None:

            return copy.deepcopy(
                history
            )

        limit = max(
            0,
            int(limit),
        )

        if limit == 0:

            return []

        return copy.deepcopy(
            history[
                -limit:
            ]
        )

    # ========================================================================
    # CLEAR HISTORY
    # ========================================================================

    def clear_history(
        self,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Clear conversation history.
        """

        session = self.require_session(
            session_id
        )

        session.history.clear()

        session.touch()

    # ========================================================================
    # SET VARIABLE
    # ========================================================================

    def set_variable(
        self,
        key: str,
        value: Any,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Set a session variable.
        """

        session = self.require_session(
            session_id
        )

        session.set_variable(
            key,
            value,
        )

    # ========================================================================
    # GET VARIABLE
    # ========================================================================

    def get_variable(
        self,
        key: str,
        default: Any = None,
        *,
        session_id: Optional[str] = None,
    ) -> Any:
        """
        Get a session variable.
        """

        session = self.require_session(
            session_id
        )

        return session.get_variable(
            key,
            default,
        )

    # ========================================================================
    # DELETE VARIABLE
    # ========================================================================

    def delete_variable(
        self,
        key: str,
        *,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Delete a session variable.
        """

        session = self.require_session(
            session_id
        )

        return session.delete_variable(
            key
        )

    # ========================================================================
    # SET CONTEXT
    # ========================================================================

    def set_context(
        self,
        key: str,
        value: Any,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Store contextual information.
        """

        session = self.require_session(
            session_id
        )

        session.set_context(
            key,
            value,
        )

    # ========================================================================
    # GET CONTEXT
    # ========================================================================

    def get_context(
        self,
        key: str,
        default: Any = None,
        *,
        session_id: Optional[str] = None,
    ) -> Any:
        """
        Retrieve contextual information.
        """

        session = self.require_session(
            session_id
        )

        return session.get_context(
            key,
            default,
        )

    # ========================================================================
    # SET METADATA
    # ========================================================================

    def set_metadata(
        self,
        key: str,
        value: Any,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Set session metadata.
        """

        session = self.require_session(
            session_id
        )

        session.metadata[
            key
        ] = copy.deepcopy(
            value
        )

        session.touch()

    # ========================================================================
    # GET METADATA
    # ========================================================================

    def get_metadata(
        self,
        key: str,
        default: Any = None,
        *,
        session_id: Optional[str] = None,
    ) -> Any:
        """
        Get session metadata.
        """

        session = self.require_session(
            session_id
        )

        session.touch()

        return copy.deepcopy(
            session.metadata.get(
                key,
                default,
            )
        )

    # ========================================================================
    # RENAME SESSION
    # ========================================================================

    def rename_session(
        self,
        name: str,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Rename a session.
        """

        session = self.require_session(
            session_id
        )

        session.name = name

        session.touch()

    # ========================================================================
    # CLEANUP EXPIRED
    # ========================================================================

    def cleanup_expired(
        self,
    ) -> int:
        """
        Remove expired sessions.
        """

        expired_ids = [
            session_id
            for session_id, session
            in self.sessions.items()
            if session.expired()
        ]

        removed = 0

        for session_id in expired_ids:

            if self.close_session(
                session_id
            ):

                removed += 1

        return removed

    # ========================================================================
    # REMOVE OLDEST
    # ========================================================================

    def _remove_oldest_session(
        self,
    ) -> None:
        """
        Remove the least recently active session.
        """

        if not self.sessions:

            return

        oldest = min(
            self.sessions.values(),
            key=lambda session:
            session.last_activity,
        )

        self.close_session(
            oldest.session_id
        )

    # ========================================================================
    # ACTIVATE LATEST
    # ========================================================================

    def _activate_latest_session(
        self,
    ) -> None:
        """
        Activate the most recently active remaining session.
        """

        if not self.sessions:

            self.active_session_id = None

            return

        latest = max(
            self.sessions.values(),
            key=lambda session:
            session.last_activity,
        )

        latest.activate()

        self.active_session_id = (
            latest.session_id
        )

    # ========================================================================
    # EXPORT SESSION
    # ========================================================================

    def export_session(
        self,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Export one session.
        """

        session = self.require_session(
            session_id
        )

        return session.to_dict()

    # ========================================================================
    # IMPORT SESSION
    # ========================================================================

    def import_session(
        self,
        data: dict[str, Any],
        *,
        activate: bool = False,
    ) -> Session:
        """
        Import a session from a dictionary.
        """

        if not isinstance(
            data,
            dict,
        ):

            raise TypeError(
                "Session data must be a dictionary."
            )

        session = Session.from_dict(
            data
        )

        if session.session_id in self.sessions:

            raise ValueError(
                "Session already exists: "
                f"{session.session_id}"
            )

        if len(
            self.sessions
        ) >= self.max_sessions:

            self._remove_oldest_session()

        self.sessions[
            session.session_id
        ] = session

        if activate:

            self.switch_session(
                session.session_id
            )

        return session

    # ========================================================================
    # EXPORT ALL
    # ========================================================================

    def export_all(
        self,
    ) -> dict[str, Any]:
        """
        Export all sessions.
        """

        return {
            "active_session_id": (
                self.active_session_id
            ),
            "sessions": {
                session_id:
                session.to_dict()
                for session_id, session
                in self.sessions.items()
            },
        }

    # ========================================================================
    # IMPORT ALL
    # ========================================================================

    def import_all(
        self,
        data: dict[str, Any],
    ) -> None:
        """
        Import all sessions.
        """

        if not isinstance(
            data,
            dict,
        ):

            raise TypeError(
                "Data must be a dictionary."
            )

        self.sessions.clear()

        sessions = data.get(
            "sessions",
            {},
        )

        if isinstance(
            sessions,
            list,
        ):

            for item in sessions:

                session = Session.from_dict(
                    item
                )

                self.sessions[
                    session.session_id
                ] = session

        elif isinstance(
            sessions,
            dict,
        ):

            for item in sessions.values():

                session = Session.from_dict(
                    item
                )

                self.sessions[
                    session.session_id
                ] = session

        active_id = data.get(
            "active_session_id"
        )

        if (
            active_id in self.sessions
        ):

            self.switch_session(
                active_id
            )

        else:

            self.active_session_id = None

    # ========================================================================
    # CLEAR ALL
    # ========================================================================

    def clear_all(
        self,
    ) -> None:
        """
        Remove all sessions.
        """

        for session in self.sessions.values():

            session.deactivate()

        self.sessions.clear()

        self.active_session_id = None

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return Session Manager statistics.
        """

        self.cleanup_expired()

        active_count = sum(
            1
            for session
            in self.sessions.values()
            if session.active
        )

        return {
            "initialized": (
                self.initialized
            ),
            "session_count": len(
                self.sessions
            ),
            "active_session_id": (
                self.active_session_id
            ),
            "active_sessions": (
                active_count
            ),
            "total_created": (
                self.total_created
            ),
            "total_closed": (
                self.total_closed
            ),
            "total_switched": (
                self.total_switched
            ),
            "total_messages": (
                self.total_messages
            ),
            "max_sessions": (
                self.max_sessions
            ),
            "started_at": (
                self.started_at
            ),
        }

    # ========================================================================
    # DEBUG INFORMATION
    # ========================================================================

    def debug_dump(
        self,
    ) -> dict[str, Any]:
        """
        Return complete debugging information.
        """

        return {
            "active_session_id": (
                self.active_session_id
            ),
            "sessions": {
                session_id:
                session.to_dict()
                for session_id, session
                in self.sessions.items()
            },
            "statistics": (
                self.statistics()
            ),
        }


# ============================================================================
# GLOBAL SESSION MANAGER
# ============================================================================

session_manager = SessionManager()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_session(
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    name: Optional[str] = None,
    metadata: Optional[
        dict[str, Any]
    ] = None,
    timeout: Optional[float] = None,
    activate: bool = True,
) -> Session:
    """
    Create a session using the global manager.
    """

    return session_manager.create_session(
        session_id=session_id,
        user_id=user_id,
        name=name,
        metadata=metadata,
        timeout=timeout,
        activate=activate,
    )


def get_session(
    session_id: Optional[str] = None,
) -> Optional[Session]:
    """
    Get a session using the global manager.
    """

    return session_manager.get_session(
        session_id
    )


def get_active_session() -> Optional[Session]:
    """
    Get the active session.
    """

    return session_manager.get_active_session()


def switch_session(
    session_id: str,
) -> Session:
    """
    Switch the active session.
    """

    return session_manager.switch_session(
        session_id
    )


def close_session(
    session_id: Optional[str] = None,
) -> bool:
    """
    Close a session.
    """

    return session_manager.close_session(
        session_id
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "Session",
    "RENIXSession",
    "SessionManager",
    "session_manager",
    "create_session",
    "get_session",
    "get_active_session",
    "switch_session",
    "close_session",
]


RENIXSession = Session
