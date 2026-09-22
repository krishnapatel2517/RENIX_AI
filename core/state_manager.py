"""
RENIX AI
Core State Manager

Central runtime state management for RENIX.

Responsibilities:
- Store runtime state
- Read/write state safely
- Nested state access
- State updates
- State deletion
- State snapshots
- State history
- State watchers
- Async watcher support
- Temporary state
- State reset
- State export/import
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import time
import uuid

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


logger = logging.getLogger(
    "RENIX.StateManager"
)


# ============================================================================
# TYPES
# ============================================================================

StateWatcher = Callable[
    [str, Any, Any],
    Any,
]


# ============================================================================
# STATE CHANGE
# ============================================================================

@dataclass
class StateChange:
    """
    Represents one state modification.
    """

    key: str

    old_value: Any

    new_value: Any

    change_id: str = field(
        default_factory=lambda:
        f"change_{uuid.uuid4().hex}"
    )

    timestamp: float = field(
        default_factory=time.time
    )

    source: Optional[str] = None

    operation: str = "update"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the state change into a dictionary.
        """

        return {
            "change_id": self.change_id,
            "key": self.key,
            "old_value": copy.deepcopy(
                self.old_value
            ),
            "new_value": copy.deepcopy(
                self.new_value
            ),
            "timestamp": self.timestamp,
            "source": self.source,
            "operation": self.operation,
            "metadata": copy.deepcopy(
                self.metadata
            ),
        }


# ============================================================================
# TEMPORARY STATE
# ============================================================================

@dataclass
class TemporaryState:
    """
    Represents a temporary state value with expiration.
    """

    key: str

    value: Any

    expires_at: float

    created_at: float = field(
        default_factory=time.time
    )

    def expired(self) -> bool:
        """
        Check whether the temporary value has expired.
        """

        return time.time() >= self.expires_at


# ============================================================================
# STATE MANAGER
# ============================================================================

class StateManager:
    """
    Central state manager for RENIX.

    Example:

        state = StateManager()

        state.set(
            "assistant.mode",
            "conversation",
        )

        mode = state.get(
            "assistant.mode"
        )
    """

    def __init__(
        self,
        *,
        max_history: int = 1000,
    ) -> None:

        self.logger = logger

        self.max_history = max(
            1,
            int(max_history),
        )

        self._state: dict[str, Any] = {}

        self._temporary: dict[
            str,
            TemporaryState,
        ] = {}

        self._watchers: dict[
            str,
            dict[
                str,
                StateWatcher,
            ],
        ] = {}

        self._global_watchers: dict[
            str,
            StateWatcher,
        ] = {}

        self._history: list[
            StateChange
        ] = []

        self._snapshots: dict[
            str,
            dict[str, Any],
        ] = {}

        self.initialized = False

        self.started_at = time.time()

        self.total_reads = 0

        self.total_writes = 0

        self.total_deletes = 0

        self.total_changes = 0

        self._lock = asyncio.Lock()

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize the state manager.
        """

        if self.initialized:

            return

        self.initialized = True

        self.logger.info(
            "RENIX State Manager initialized."
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Shut down the state manager.
        """

        self.initialized = False

        self.logger.info(
            "RENIX State Manager shutdown."
        )

    # ========================================================================
    # GET
    # ========================================================================

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get a value from state.

        Supports nested keys:

            state.get("user.profile.name")
        """

        self.total_reads += 1

        self._cleanup_expired()

        if not key:

            return default

        parts = self._split_key(
            key
        )

        current: Any = self._state

        for part in parts:

            if not isinstance(
                current,
                dict,
            ):

                return default

            if part not in current:

                return default

            current = current[
                part
            ]

        return copy.deepcopy(
            current
        )

    # ========================================================================
    # HAS
    # ========================================================================

    def has(
        self,
        key: str,
    ) -> bool:
        """
        Check whether a state key exists.
        """

        sentinel = object()

        return self.get(
            key,
            sentinel,
        ) is not sentinel

    # ========================================================================
    # SET
    # ========================================================================

    def set(
        self,
        key: str,
        value: Any,
        *,
        source: Optional[str] = None,
        notify: bool = True,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Any:
        """
        Set a state value.

        Nested keys are automatically created.
        """

        self._validate_key(
            key
        )

        old_value = self.get(
            key,
            None,
        )

        parts = self._split_key(
            key
        )

        current = self._state

        for part in parts[:-1]:

            existing = current.get(
                part
            )

            if not isinstance(
                existing,
                dict,
            ):

                existing = {}

                current[
                    part
                ] = existing

            current = existing

        current[
            parts[-1]
        ] = copy.deepcopy(
            value
        )

        self.total_writes += 1

        changed = (
            old_value != value
        )

        if changed:

            self.total_changes += 1

            change = StateChange(
                key=key,
                old_value=old_value,
                new_value=value,
                source=source,
                operation="update",
                metadata=dict(
                    metadata or {}
                ),
            )

            self._record_change(
                change
            )

            if notify:

                self._notify_watchers(
                    change
                )

        return copy.deepcopy(
            value
        )

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        values: dict[str, Any],
        *,
        source: Optional[str] = None,
        notify: bool = True,
    ) -> None:
        """
        Update multiple state values.
        """

        if not isinstance(
            values,
            dict,
        ):

            raise TypeError(
                "values must be a dictionary."
            )

        for key, value in values.items():

            self.set(
                key,
                value,
                source=source,
                notify=notify,
            )

    # ========================================================================
    # SET DEFAULT
    # ========================================================================

    def set_default(
        self,
        key: str,
        value: Any,
        *,
        source: Optional[str] = None,
    ) -> Any:
        """
        Set a value only if the key does not already exist.
        """

        if self.has(key):

            return self.get(
                key
            )

        return self.set(
            key,
            value,
            source=source,
        )

    # ========================================================================
    # DELETE
    # ========================================================================

    def delete(
        self,
        key: str,
        *,
        source: Optional[str] = None,
        notify: bool = True,
    ) -> bool:
        """
        Delete a state key.
        """

        self._validate_key(
            key
        )

        parts = self._split_key(
            key
        )

        current = self._state

        for part in parts[:-1]:

            if not isinstance(
                current,
                dict,
            ):

                return False

            if part not in current:

                return False

            current = current[
                part
            ]

        final_key = parts[-1]

        if final_key not in current:

            return False

        old_value = copy.deepcopy(
            current[
                final_key
            ]
        )

        del current[
            final_key
        ]

        self.total_deletes += 1

        self.total_changes += 1

        change = StateChange(
            key=key,
            old_value=old_value,
            new_value=None,
            source=source,
            operation="delete",
        )

        self._record_change(
            change
        )

        if notify:

            self._notify_watchers(
                change
            )

        return True

    # ========================================================================
    # CLEAR
    # ========================================================================

    def clear(
        self,
        *,
        source: Optional[str] = None,
        notify: bool = True,
    ) -> None:
        """
        Clear all runtime state.
        """

        old_state = copy.deepcopy(
            self._state
        )

        self._state.clear()

        if old_state:

            self.total_changes += 1

            change = StateChange(
                key="*",
                old_value=old_state,
                new_value={},
                source=source,
                operation="clear",
            )

            self._record_change(
                change
            )

            if notify:

                self._notify_watchers(
                    change
                )

    # ========================================================================
    # RESET
    # ========================================================================

    def reset(
        self,
        initial_state: Optional[
            dict[str, Any]
        ] = None,
        *,
        source: Optional[str] = None,
    ) -> None:
        """
        Completely reset runtime state.
        """

        self._state.clear()

        self._temporary.clear()

        self._snapshots.clear()

        if initial_state:

            self._state = copy.deepcopy(
                initial_state
            )

        self.logger.info(
            "RENIX state reset."
        )

    # ========================================================================
    # NESTED GET
    # ========================================================================

    def get_nested(
        self,
        *keys: str,
        default: Any = None,
    ) -> Any:
        """
        Read nested state using multiple key arguments.
        """

        if not keys:

            return default

        return self.get(
            ".".join(keys),
            default,
        )

    # ========================================================================
    # NESTED SET
    # ========================================================================

    def set_nested(
        self,
        keys: list[str] | tuple[str, ...],
        value: Any,
        *,
        source: Optional[str] = None,
    ) -> Any:
        """
        Set nested state using multiple keys.
        """

        if not keys:

            raise ValueError(
                "keys cannot be empty."
            )

        return self.set(
            ".".join(keys),
            value,
            source=source,
        )

    # ========================================================================
    # APPEND
    # ========================================================================

    def append(
        self,
        key: str,
        value: Any,
        *,
        source: Optional[str] = None,
    ) -> list[Any]:
        """
        Append a value to a list stored in state.
        """

        current = self.get(
            key
        )

        if current is None:

            current = []

        if not isinstance(
            current,
            list,
        ):

            raise TypeError(
                f"State '{key}' is not a list."
            )

        current.append(
            copy.deepcopy(
                value
            )
        )

        self.set(
            key,
            current,
            source=source,
        )

        return current

    # ========================================================================
    # REMOVE FROM LIST
    # ========================================================================

    def remove_from_list(
        self,
        key: str,
        value: Any,
        *,
        source: Optional[str] = None,
    ) -> bool:
        """
        Remove a value from a state list.
        """

        current = self.get(
            key
        )

        if not isinstance(
            current,
            list,
        ):

            return False

        try:

            current.remove(
                value
            )

        except ValueError:

            return False

        self.set(
            key,
            current,
            source=source,
        )

        return True

    # ========================================================================
    # TEMPORARY SET
    # ========================================================================

    def set_temporary(
        self,
        key: str,
        value: Any,
        ttl: float,
        *,
        source: Optional[str] = None,
    ) -> Any:
        """
        Set a temporary value that expires after ttl seconds.
        """

        if ttl <= 0:

            raise ValueError(
                "ttl must be greater than zero."
            )

        result = self.set(
            key,
            value,
            source=source,
        )

        self._temporary[
            key
        ] = TemporaryState(
            key=key,
            value=copy.deepcopy(
                value
            ),
            expires_at=(
                time.time()
                + ttl
            ),
        )

        return result

    # ========================================================================
    # TEMPORARY GET
    # ========================================================================

    def get_temporary(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get a temporary value.
        """

        temporary = self._temporary.get(
            key
        )

        if temporary is None:

            return default

        if temporary.expired():

            self._temporary.pop(
                key,
                None,
            )

            self.delete(
                key,
                notify=False,
            )

            return default

        return copy.deepcopy(
            temporary.value
        )

    # ========================================================================
    # CLEANUP EXPIRED
    # ========================================================================

    def _cleanup_expired(
        self,
    ) -> None:
        """
        Remove expired temporary values.
        """

        expired = [
            key
            for key, item
            in self._temporary.items()
            if item.expired()
        ]

        for key in expired:

            self._temporary.pop(
                key,
                None,
            )

            self.delete(
                key,
                notify=False,
            )

    # ========================================================================
    # WATCH
    # ========================================================================

    def watch(
        self,
        key: str,
        callback: StateWatcher,
    ) -> str:
        """
        Watch a specific state key.

        Callback receives:

            key
            old_value
            new_value
        """

        self._validate_key(
            key
        )

        if not callable(
            callback
        ):

            raise TypeError(
                "callback must be callable."
            )

        watcher_id = (
            f"watcher_{uuid.uuid4().hex}"
        )

        self._watchers.setdefault(
            key,
            {},
        )[
            watcher_id
        ] = callback

        return watcher_id

    # ========================================================================
    # WATCH ALL
    # ========================================================================

    def watch_all(
        self,
        callback: StateWatcher,
    ) -> str:
        """
        Watch every state change.
        """

        if not callable(
            callback
        ):

            raise TypeError(
                "callback must be callable."
            )

        watcher_id = (
            f"watcher_{uuid.uuid4().hex}"
        )

        self._global_watchers[
            watcher_id
        ] = callback

        return watcher_id

    # ========================================================================
    # UNWATCH
    # ========================================================================

    def unwatch(
        self,
        watcher_id: str,
    ) -> bool:
        """
        Remove a watcher.
        """

        if self._global_watchers.pop(
            watcher_id,
            None,
        ) is not None:

            return True

        for watchers in self._watchers.values():

            if watchers.pop(
                watcher_id,
                None,
            ) is not None:

                return True

        return False

    # ========================================================================
    # NOTIFY WATCHERS
    # ========================================================================

    def _notify_watchers(
        self,
        change: StateChange,
    ) -> None:
        """
        Notify registered watchers.

        Async callbacks are scheduled automatically
        when an active event loop exists.
        """

        callbacks: list[
            StateWatcher
        ] = []

        callbacks.extend(
            self._global_watchers.values()
        )

        callbacks.extend(
            self._watchers.get(
                change.key,
                {},
            ).values()
        )

        for callback in callbacks:

            try:

                result = callback(
                    change.key,
                    copy.deepcopy(
                        change.old_value
                    ),
                    copy.deepcopy(
                        change.new_value
                    ),
                )

                if inspect.isawaitable(
                    result
                ):

                    try:

                        loop = (
                            asyncio.get_running_loop()
                        )

                        loop.create_task(
                            result
                        )

                    except RuntimeError:

                        pass

            except Exception:

                self.logger.exception(
                    "State watcher failed."
                )

    # ========================================================================
    # HISTORY
    # ========================================================================

    def _record_change(
        self,
        change: StateChange,
    ) -> None:
        """
        Record a state change.
        """

        self._history.append(
            change
        )

        if len(
            self._history
        ) > self.max_history:

            self._history = self._history[
                -self.max_history:
            ]

    # ========================================================================
    # GET HISTORY
    # ========================================================================

    def get_history(
        self,
        limit: Optional[int] = None,
        *,
        key: Optional[str] = None,
    ) -> list[StateChange]:
        """
        Return state change history.
        """

        history = self._history

        if key is not None:

            history = [
                change
                for change in history
                if change.key == key
            ]

        if limit is None:

            return list(
                history
            )

        limit = max(
            0,
            int(limit),
        )

        if limit == 0:

            return []

        return history[
            -limit:
        ]

    # ========================================================================
    # CLEAR HISTORY
    # ========================================================================

    def clear_history(
        self,
    ) -> None:
        """
        Clear state change history.
        """

        self._history.clear()

    # ========================================================================
    # SNAPSHOT
    # ========================================================================

    def create_snapshot(
        self,
        name: Optional[str] = None,
    ) -> str:
        """
        Create a snapshot of current state.
        """

        snapshot_id = (
            name
            or f"snapshot_{uuid.uuid4().hex}"
        )

        self._snapshots[
            snapshot_id
        ] = copy.deepcopy(
            self._state
        )

        return snapshot_id

    # ========================================================================
    # RESTORE SNAPSHOT
    # ========================================================================

    def restore_snapshot(
        self,
        snapshot_id: str,
        *,
        source: Optional[str] = None,
    ) -> bool:
        """
        Restore a previously created snapshot.
        """

        snapshot = self._snapshots.get(
            snapshot_id
        )

        if snapshot is None:

            return False

        old_state = copy.deepcopy(
            self._state
        )

        self._state = copy.deepcopy(
            snapshot
        )

        change = StateChange(
            key="*",
            old_value=old_state,
            new_value=copy.deepcopy(
                self._state
            ),
            source=source,
            operation="restore_snapshot",
            metadata={
                "snapshot_id": snapshot_id
            },
        )

        self.total_changes += 1

        self._record_change(
            change
        )

        self._notify_watchers(
            change
        )

        return True

    # ========================================================================
    # DELETE SNAPSHOT
    # ========================================================================

    def delete_snapshot(
        self,
        snapshot_id: str,
    ) -> bool:
        """
        Delete a snapshot.
        """

        if snapshot_id not in self._snapshots:

            return False

        del self._snapshots[
            snapshot_id
        ]

        return True

    # ========================================================================
    # GET SNAPSHOTS
    # ========================================================================

    def get_snapshots(
        self,
    ) -> list[str]:
        """
        Return snapshot names.
        """

        return list(
            self._snapshots.keys()
        )

    # ========================================================================
    # EXPORT
    # ========================================================================

    def export_state(
        self,
    ) -> dict[str, Any]:
        """
        Export current state.
        """

        self._cleanup_expired()

        return copy.deepcopy(
            self._state
        )

    # ========================================================================
    # IMPORT
    # ========================================================================

    def import_state(
        self,
        state: dict[str, Any],
        *,
        replace: bool = True,
        source: Optional[str] = None,
    ) -> None:
        """
        Import external state.
        """

        if not isinstance(
            state,
            dict,
        ):

            raise TypeError(
                "state must be a dictionary."
            )

        if replace:

            old_state = copy.deepcopy(
                self._state
            )

            self._state = copy.deepcopy(
                state
            )

            change = StateChange(
                key="*",
                old_value=old_state,
                new_value=copy.deepcopy(
                    self._state
                ),
                source=source,
                operation="import",
            )

            self.total_changes += 1

            self._record_change(
                change
            )

            self._notify_watchers(
                change
            )

            return

        self.update(
            state,
            source=source,
        )

    # ========================================================================
    # STATE COPY
    # ========================================================================

    def copy(
        self,
    ) -> dict[str, Any]:
        """
        Return a deep copy of current state.
        """

        return copy.deepcopy(
            self._state
        )

    # ========================================================================
    # KEYS
    # ========================================================================

    def keys(
        self,
    ) -> list[str]:
        """
        Return top-level state keys.
        """

        return list(
            self._state.keys()
        )

    # ========================================================================
    # VALUES
    # ========================================================================

    def values(
        self,
    ) -> list[Any]:
        """
        Return top-level state values.
        """

        return [
            copy.deepcopy(value)
            for value
            in self._state.values()
        ]

    # ========================================================================
    # ITEMS
    # ========================================================================

    def items(
        self,
    ) -> list[
        tuple[str, Any]
    ]:
        """
        Return top-level state items.
        """

        return [
            (
                key,
                copy.deepcopy(
                    value
                ),
            )
            for key, value
            in self._state.items()
        ]

    # ========================================================================
    # SIZE
    # ========================================================================

    def size(
        self,
    ) -> int:
        """
        Return number of top-level state entries.
        """

        return len(
            self._state
        )

    # ========================================================================
    # VALIDATE KEY
    # ========================================================================

    @staticmethod
    def _validate_key(
        key: str,
    ) -> None:
        """
        Validate a state key.
        """

        if not isinstance(
            key,
            str,
        ):

            raise TypeError(
                "State key must be a string."
            )

        if not key.strip():

            raise ValueError(
                "State key cannot be empty."
            )

    # ========================================================================
    # SPLIT KEY
    # ========================================================================

    @staticmethod
    def _split_key(
        key: str,
    ) -> list[str]:
        """
        Split a dotted state key.
        """

        return [
            part.strip()
            for part in key.split(".")
            if part.strip()
        ]

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return State Manager statistics.
        """

        self._cleanup_expired()

        return {
            "initialized": (
                self.initialized
            ),
            "state_size": len(
                self._state
            ),
            "temporary_values": len(
                self._temporary
            ),
            "watcher_count": (
                len(
                    self._global_watchers
                )
                + sum(
                    len(
                        watchers
                    )
                    for watchers
                    in self._watchers.values()
                )
            ),
            "snapshot_count": len(
                self._snapshots
            ),
            "history_count": len(
                self._history
            ),
            "total_reads": (
                self.total_reads
            ),
            "total_writes": (
                self.total_writes
            ),
            "total_deletes": (
                self.total_deletes
            ),
            "total_changes": (
                self.total_changes
            ),
            "started_at": (
                self.started_at
            ),
        }

    # ========================================================================
    # DEBUG DUMP
    # ========================================================================

    def debug_dump(
        self,
    ) -> dict[str, Any]:
        """
        Return detailed State Manager information.
        """

        return {
            "state": self.export_state(),
            "temporary": {
                key: {
                    "expires_at": item.expires_at,
                    "created_at": item.created_at,
                }
                for key, item
                in self._temporary.items()
            },
            "snapshots": list(
                self._snapshots.keys()
            ),
            "history": [
                change.to_dict()
                for change in self._history
            ],
            "statistics": (
                self.statistics()
            ),
        }


# ============================================================================
# GLOBAL STATE MANAGER
# ============================================================================

state_manager = StateManager()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get(
    key: str,
    default: Any = None,
) -> Any:
    """
    Get a value from the global State Manager.
    """

    return state_manager.get(
        key,
        default,
    )


def set(
    key: str,
    value: Any,
    *,
    source: Optional[str] = None,
) -> Any:
    """
    Set a value in the global State Manager.
    """

    return state_manager.set(
        key,
        value,
        source=source,
    )


def update(
    values: dict[str, Any],
    *,
    source: Optional[str] = None,
) -> None:
    """
    Update global state.
    """

    state_manager.update(
        values,
        source=source,
    )


def delete(
    key: str,
    *,
    source: Optional[str] = None,
) -> bool:
    """
    Delete a global state key.
    """

    return state_manager.delete(
        key,
        source=source,
    )


def has(
    key: str,
) -> bool:
    """
    Check global state.
    """

    return state_manager.has(
        key
    )


def watch(
    key: str,
    callback: StateWatcher,
) -> str:
    """
    Watch a global state key.
    """

    return state_manager.watch(
        key,
        callback,
    )


def watch_all(
    callback: StateWatcher,
) -> str:
    """
    Watch all global state changes.
    """

    return state_manager.watch_all(
        callback
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "StateChange",
    "TemporaryState",
    "StateManager",
    "state_manager",
    "get",
    "set",
    "update",
    "delete",
    "has",
    "watch",
    "watch_all",
]


