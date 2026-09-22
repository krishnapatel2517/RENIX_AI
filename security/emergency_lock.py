"""
RENIX Emergency Lock
====================

Provides an emergency security lock for RENIX.

Features:

* Immediate emergency lock activation
* Multiple lock levels
* Session locking
* Sensitive operation blocking
* Emergency unlock with authorization
* Optional cooldown period
* Lock history
* Event callbacks
* Integration with SecureMode and other security modules

This module is intended as a centralized application-level
emergency lock controller.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("RENIX.EmergencyLock")

class EmergencyLock:
    """
    Emergency security lock controller for RENIX.

    Lock levels:

    soft
        Blocks sensitive operations.

    hard
        Blocks most operations and locks the session.

    critical
        Full application emergency lock. Only explicitly
        allowed recovery operations should be available.
    """

    VALID_LEVELS = {
        "soft",
        "hard",
        "critical",
    }

    LEVEL_PRIORITIES = {
        "soft": 1,
        "hard": 2,
        "critical": 3,
    }

    DEFAULT_BLOCKED_OPERATIONS = {
        "soft": {
            "delete_file",
            "delete_files",
            "system_change",
            "device_control",
            "automation",
        },

        "hard": {
            "delete_file",
            "delete_files",
            "system_change",
            "device_control",
            "automation",
            "shell_command",
            "terminal",
            "external_integration",
            "network_request",
        },

        "critical": {
            "*",
        },
    }

    DEFAULT_ALLOWED_DURING_LOCK = {
        "status",
        "emergency_status",
        "unlock_request",
        "security_recovery",
        "shutdown",
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize Emergency Lock.
        """

        self.config = config or {}

        self.locked = False

        self.level: Optional[str] = None

        self.reason: Optional[str] = None

        self.locked_at: Optional[
            datetime
        ] = None

        self.expires_at: Optional[
            datetime
        ] = None

        self.locked_by: Optional[
            str
        ] = None

        self.session_locked = False

        self.require_authorization = bool(
            self.config.get(
                "require_authorization",
                True,
            )
        )

        self.cooldown_seconds = int(
            self.config.get(
                "cooldown_seconds",
                0,
            )
        )

        self.allowed_during_lock = set(
            self.config.get(
                "allowed_during_lock",
                self.DEFAULT_ALLOWED_DURING_LOCK,
            )
        )

        self.blocked_operations = {
            level: set(operations)
            for (
                level,
                operations,
            ) in (
                self.DEFAULT_BLOCKED_OPERATIONS
                .items()
            )
        }

        self.history: List[
            Dict[str, Any]
        ] = []

        self.max_history = int(
            self.config.get(
                "max_history",
                500,
            )
        )

        self.event_handlers: List[
            Callable[
                [Dict[str, Any]],
                None,
            ]
        ] = []

        self._lock = threading.RLock()

        self._load_custom_config()

        logger.info(
            "EmergencyLock initialized."
        )

    # ============================================================
    # CONFIGURATION
    # ============================================================

    def _load_custom_config(
        self,
    ) -> None:
        """
        Load custom blocked operations.
        """

        custom_operations = (
            self.config.get(
                "blocked_operations",
                {},
            )
        )

        if not isinstance(
            custom_operations,
            dict,
        ):
            return

        for (
            level,
            operations,
        ) in custom_operations.items():

            level = (
                str(level)
                .lower()
                .strip()
            )

            if level not in (
                self.VALID_LEVELS
            ):
                continue

            if not isinstance(
                operations,
                (
                    list,
                    set,
                    tuple,
                ),
            ):
                continue

            self.blocked_operations[
                level
            ] = set(
                str(operation)
                .lower()
                .strip()
                for operation in operations
            )

    # ============================================================
    # LOCK ACTIVATION
    # ============================================================

    def activate(
        self,
        level: str = "hard",
        reason: Optional[str] = None,
        locked_by: Optional[str] = None,
        duration: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Activate emergency lock.

        Args:
            level:
                soft, hard, or critical.

            reason:
                Reason for emergency lock.

            locked_by:
                User or system that activated the lock.

            duration:
                Optional automatic expiration in seconds.

            force:
                Allows replacing a stronger lock level.
        """

        level = self._normalize_level(
            level
        )

        with self._lock:

            self._check_expiration()

            if (
                self.locked
                and self.level is not None
                and not force
            ):

                current_priority = (
                    self.LEVEL_PRIORITIES[
                        self.level
                    ]
                )

                requested_priority = (
                    self.LEVEL_PRIORITIES[
                        level
                    ]
                )

                if (
                    requested_priority
                    < current_priority
                ):

                    return self.get_status()

            self.locked = True

            self.level = level

            self.reason = (
                reason
                or "Emergency lock activated."
            )

            self.locked_by = locked_by

            self.locked_at = (
                datetime.now()
            )

            self.session_locked = (
                level
                in {
                    "hard",
                    "critical",
                }
            )

            if duration is not None:

                self.expires_at = (
                    datetime.now()
                    + timedelta(
                        seconds=max(
                            1,
                            int(duration),
                        )
                    )
                )

            else:

                self.expires_at = None

            event = {
                "event": (
                    "emergency_lock_activated"
                ),
                "level": level,
                "reason": self.reason,
                "locked_by": locked_by,
                "timestamp": (
                    datetime.now().isoformat()
                ),
                "expires_at": (
                    self.expires_at.isoformat()
                    if self.expires_at
                    else None
                ),
            }

            self._add_history(
                event
            )

        logger.critical(
            "EMERGENCY LOCK ACTIVATED | "
            "Level: %s | Reason: %s",
            level,
            self.reason,
        )

        self._notify(
            event
        )

        return self.get_status()

    # ============================================================
    # CONVENIENCE LOCK METHODS
    # ============================================================

    def soft_lock(
        self,
        reason: Optional[str] = None,
        locked_by: Optional[
            str
        ] = None,
    ) -> Dict[str, Any]:
        """
        Activate soft emergency lock.
        """

        return self.activate(
            level="soft",
            reason=reason,
            locked_by=locked_by,
        )

    def hard_lock(
        self,
        reason: Optional[str] = None,
        locked_by: Optional[
            str
        ] = None,
    ) -> Dict[str, Any]:
        """
        Activate hard emergency lock.
        """

        return self.activate(
            level="hard",
            reason=reason,
            locked_by=locked_by,
        )

    def critical_lock(
        self,
        reason: Optional[str] = None,
        locked_by: Optional[
            str
        ] = None,
    ) -> Dict[str, Any]:
        """
        Activate critical emergency lock.
        """

        return self.activate(
            level="critical",
            reason=reason,
            locked_by=locked_by,
            force=True,
        )

    # ============================================================
    # UNLOCK
    # ============================================================

    def unlock(
        self,
        authorized: bool = False,
        unlocked_by: Optional[
            str
        ] = None,
        reason: Optional[
            str
        ] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Unlock emergency lock.

        If authorization is required, authorized=True
        must be supplied by the calling authentication
        or security system.
        """

        with self._lock:

            self._check_expiration()

            if not self.locked:

                return self.get_status()

            if (
                self.require_authorization
                and not authorized
                and not force
            ):

                logger.warning(
                    "Unauthorized emergency unlock "
                    "attempt."
                )

                event = {
                    "event": (
                        "emergency_unlock_denied"
                    ),
                    "reason": (
                        "Authorization required."
                    ),
                    "timestamp": (
                        datetime.now().isoformat()
                    ),
                    "requested_by": (
                        unlocked_by
                    ),
                }

                self._add_history(
                    event
                )

                self._notify(
                    event
                )

                return self.get_status()

            if not force:

                if (
                    not self._cooldown_complete()
                ):

                    logger.warning(
                        "Emergency unlock attempted "
                        "before cooldown."
                    )

                    return self.get_status()

            previous_level = self.level

            self.locked = False

            self.level = None

            self.reason = reason

            self.locked_by = None

            self.locked_at = None

            self.expires_at = None

            self.session_locked = False

            event = {
                "event": (
                    "emergency_lock_released"
                ),
                "previous_level": (
                    previous_level
                ),
                "reason": (
                    reason
                    or "Emergency lock released."
                ),
                "unlocked_by": (
                    unlocked_by
                ),
                "timestamp": (
                    datetime.now().isoformat()
                ),
            }

            self._add_history(
                event
            )

        logger.warning(
            "Emergency lock released."
        )

        self._notify(
            event
        )

        return self.get_status()

    # ============================================================
    # COOLDOWN
    # ============================================================

    def _cooldown_complete(
        self,
    ) -> bool:
        """
        Check whether lock cooldown is complete.
        """

        if self.cooldown_seconds <= 0:

            return True

        if self.locked_at is None:

            return True

        unlock_time = (
            self.locked_at
            + timedelta(
                seconds=(
                    self.cooldown_seconds
                )
            )
        )

        return (
            datetime.now()
            >= unlock_time
        )

    # ============================================================
    # EXPIRATION
    # ============================================================

    def _check_expiration(
        self,
    ) -> bool:
        """
        Automatically release expired emergency lock.
        """

        if not self.locked:

            return False

        if self.expires_at is None:

            return False

        if datetime.now() < self.expires_at:

            return False

        previous_level = self.level

        self.locked = False

        self.level = None

        self.reason = (
            "Emergency lock expired."
        )

        self.locked_by = None

        self.locked_at = None

        self.expires_at = None

        self.session_locked = False

        event = {
            "event": (
                "emergency_lock_expired"
            ),
            "previous_level": (
                previous_level
            ),
            "timestamp": (
                datetime.now().isoformat()
            ),
        }

        self._add_history(
            event
        )

        logger.warning(
            "Emergency lock automatically expired."
        )

        self._notify(
            event
        )

        return True

    def check_expiration(
        self,
    ) -> bool:
        """
        Public expiration check.
        """

        with self._lock:

            return self._check_expiration()

    # ============================================================
    # OPERATION CHECKING
    # ============================================================

    def is_operation_allowed(
        self,
        operation: str,
    ) -> bool:
        """
        Check whether an operation is allowed
        during emergency lock.
        """

        with self._lock:

            self._check_expiration()

            if not self.locked:

                return True

            operation = (
                str(operation)
                .lower()
                .strip()
            )

            if operation in (
                self.allowed_during_lock
            ):

                return True

            if self.level is None:

                return True

            blocked = (
                self.blocked_operations.get(
                    self.level,
                    set(),
                )
            )

            if "*" in blocked:

                return False

            return (
                operation
                not in blocked
            )

    def is_session_locked(
        self,
    ) -> bool:
        """
        Check whether user session should be locked.
        """

        with self._lock:

            self._check_expiration()

            return self.session_locked

    # ============================================================
    # LEVEL
    # ============================================================

    def _normalize_level(
        self,
        level: str,
    ) -> str:
        """
        Normalize emergency lock level.
        """

        level = (
            str(level)
            .lower()
            .strip()
        )

        if level not in (
            self.VALID_LEVELS
        ):

            logger.warning(
                "Invalid emergency lock level: %s",
                level,
            )

            return "hard"

        return level

    # ============================================================
    # EVENT HANDLERS
    # ============================================================

    def add_event_handler(
        self,
        handler: Callable[
            [Dict[str, Any]],
            None,
        ],
    ) -> None:
        """
        Register emergency lock event handler.
        """

        if not callable(
            handler
        ):

            return

        with self._lock:

            if (
                handler
                not in self.event_handlers
            ):

                self.event_handlers.append(
                    handler
                )

    def remove_event_handler(
        self,
        handler: Callable,
    ) -> bool:
        """
        Remove event handler.
        """

        with self._lock:

            if (
                handler
                not in self.event_handlers
            ):

                return False

            self.event_handlers.remove(
                handler
            )

            return True

    def _notify(
        self,
        event: Dict[str, Any],
    ) -> None:
        """
        Notify registered event handlers.
        """

        handlers = list(
            self.event_handlers
        )

        for handler in handlers:

            try:

                handler(
                    event.copy()
                )

            except Exception as error:

                logger.error(
                    "Emergency lock event handler "
                    "failed: %s",
                    error,
                )

    # ============================================================
    # HISTORY
    # ============================================================

    def _add_history(
        self,
        event: Dict[str, Any],
    ) -> None:
        """
        Add event to lock history.
        """

        self.history.append(
            event
        )

        if (
            len(self.history)
            > self.max_history
        ):

            overflow = (
                len(self.history)
                - self.max_history
            )

            self.history = (
                self.history[
                    overflow:
                ]
            )

    def get_history(
        self,
        limit: int = 100,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Return emergency lock history.
        """

        with self._lock:

            return [
                event.copy()
                for event in self.history[
                    -max(
                        1,
                        limit,
                    ):
                ]
            ]

    def clear_history(
        self,
    ) -> None:
        """
        Clear emergency lock history.
        """

        with self._lock:

            self.history.clear()

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return current emergency lock status.
        """

        with self._lock:

            self._check_expiration()

            return {
                "locked": self.locked,
                "level": self.level,
                "reason": self.reason,
                "locked_by": self.locked_by,
                "locked_at": (
                    self.locked_at.isoformat()
                    if self.locked_at
                    else None
                ),
                "expires_at": (
                    self.expires_at.isoformat()
                    if self.expires_at
                    else None
                ),
                "session_locked": (
                    self.session_locked
                ),
                "authorization_required": (
                    self.require_authorization
                ),
                "cooldown_seconds": (
                    self.cooldown_seconds
                ),
                "history_entries": (
                    len(self.history)
                ),
            }

    def is_locked(
        self,
    ) -> bool:
        """
        Check whether emergency lock is active.
        """

        with self._lock:

            self._check_expiration()

            return self.locked

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Shut down EmergencyLock.
        """

        logger.info(
            "Shutting down EmergencyLock..."
        )

        with self._lock:

            self.event_handlers.clear()

        logger.info(
            "EmergencyLock shutdown complete."
        )
