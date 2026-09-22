"""
RENIX Secure Mode
=================

Provides a centralized secure-mode system for RENIX.

Secure Mode can temporarily increase security restrictions when:

* Suspicious activity is detected
* A critical threat occurs
* The user manually enables secure mode
* An emergency situation occurs

Features:

* Enable and disable secure mode
* Multiple security levels
* Restricted capabilities
* Temporary secure mode
* Automatic expiration
* Security policies
* Status monitoring
* Event callbacks
  """

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("RENIX.SecureMode")

class SecureMode:
    """
    Centralized security restriction controller.

    Security levels:

    normal
        Standard RENIX permissions.

    cautious
        Increased confirmation requirements.

    restricted
        Sensitive operations disabled.

    lockdown
        Only essential operations allowed.

    Example:

        secure_mode = SecureMode()

        secure_mode.enable(
            level="restricted",
            reason="Suspicious activity detected"
        )

        if secure_mode.is_operation_allowed(
            "delete_files"
        ):
            ...
    """

    VALID_LEVELS = {
        "normal",
        "cautious",
        "restricted",
        "lockdown",
    }

    LEVEL_PRIORITIES = {
        "normal": 0,
        "cautious": 1,
        "restricted": 2,
        "lockdown": 3,
    }

    DEFAULT_POLICIES = {
        "normal": {
            "require_confirmation": False,
            "allow_shell_commands": True,
            "allow_file_deletion": True,
            "allow_system_changes": True,
            "allow_network_requests": True,
            "allow_device_control": True,
            "allow_automation": True,
            "allow_external_integrations": True,
        },

        "cautious": {
            "require_confirmation": True,
            "allow_shell_commands": True,
            "allow_file_deletion": True,
            "allow_system_changes": True,
            "allow_network_requests": True,
            "allow_device_control": True,
            "allow_automation": True,
            "allow_external_integrations": True,
        },

        "restricted": {
            "require_confirmation": True,
            "allow_shell_commands": False,
            "allow_file_deletion": False,
            "allow_system_changes": False,
            "allow_network_requests": True,
            "allow_device_control": False,
            "allow_automation": False,
            "allow_external_integrations": False,
        },

        "lockdown": {
            "require_confirmation": True,
            "allow_shell_commands": False,
            "allow_file_deletion": False,
            "allow_system_changes": False,
            "allow_network_requests": False,
            "allow_device_control": False,
            "allow_automation": False,
            "allow_external_integrations": False,
        },
    }

    OPERATION_POLICY_MAP = {
        "shell_command": "allow_shell_commands",
        "terminal": "allow_shell_commands",
        "delete_file": "allow_file_deletion",
        "delete_files": "allow_file_deletion",
        "system_change": "allow_system_changes",
        "system_settings": "allow_system_changes",
        "network_request": "allow_network_requests",
        "device_control": "allow_device_control",
        "automation": "allow_automation",
        "external_integration": (
            "allow_external_integrations"
        ),
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize Secure Mode.
        """

        self.config = config or {}

        self.enabled = False

        self.current_level = "normal"

        self.reason: Optional[str] = None

        self.enabled_at: Optional[
            datetime
        ] = None

        self.expires_at: Optional[
            datetime
        ] = None

        self.policies = {
            level: dict(policy)
            for level, policy in (
                self.DEFAULT_POLICIES.items()
            )
        }

        self.event_handlers: List[
            Callable[[Dict[str, Any]], None]
        ] = []

        self.history: List[
            Dict[str, Any]
        ] = []

        self.max_history = int(
            self.config.get(
                "max_history",
                500,
            )
        )

        self._lock = threading.RLock()

        self._load_custom_policies()

        logger.info(
            "SecureMode initialized."
        )

    # ============================================================
    # CONFIGURATION
    # ============================================================

    def _load_custom_policies(
        self,
    ) -> None:
        """
        Load custom security policies from config.
        """

        custom_policies = (
            self.config.get(
                "policies",
                {},
            )
        )

        if not isinstance(
            custom_policies,
            dict,
        ):
            return

        for (
            level,
            policy,
        ) in custom_policies.items():

            level = (
                str(level)
                .lower()
                .strip()
            )

            if level not in self.VALID_LEVELS:
                continue

            if not isinstance(
                policy,
                dict,
            ):
                continue

            self.policies[
                level
            ].update(
                policy
            )

    # ============================================================
    # ENABLE SECURE MODE
    # ============================================================

    def enable(
        self,
        level: str = "restricted",
        reason: Optional[str] = None,
        duration: Optional[
            int
        ] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Enable secure mode.

        Args:
            level:
                normal, cautious, restricted,
                or lockdown.

            reason:
                Reason secure mode was enabled.

            duration:
                Optional duration in seconds.

            force:
                Allows lowering restrictions when secure
                mode is already active.
        """

        level = self._normalize_level(
            level
        )

        with self._lock:

            self._check_expiration()

            if (
                self.enabled
                and not force
            ):

                current_priority = (
                    self.LEVEL_PRIORITIES[
                        self.current_level
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

            self.enabled = (
                level != "normal"
            )

            self.current_level = level

            self.reason = reason

            self.enabled_at = (
                datetime.now()
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
                    "secure_mode_enabled"
                    if self.enabled
                    else "secure_mode_normal"
                ),
                "level": level,
                "reason": reason,
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

        logger.warning(
            "Secure mode changed to: %s",
            level,
        )

        self._notify(
            event
        )

        return self.get_status()

    # ============================================================
    # DISABLE SECURE MODE
    # ============================================================

    def disable(
        self,
        reason: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Disable secure mode and return to normal.

        Lockdown mode requires force=True unless
        explicitly expired.
        """

        with self._lock:

            self._check_expiration()

            if (
                self.current_level == "lockdown"
                and not force
            ):

                logger.warning(
                    "Attempt to disable lockdown "
                    "without force."
                )

                return self.get_status()

            previous_level = (
                self.current_level
            )

            self.enabled = False

            self.current_level = "normal"

            self.reason = reason

            self.enabled_at = None

            self.expires_at = None

            event = {
                "event": (
                    "secure_mode_disabled"
                ),
                "previous_level": (
                    previous_level
                ),
                "reason": reason,
                "timestamp": (
                    datetime.now().isoformat()
                ),
            }

            self._add_history(
                event
            )

        logger.info(
            "Secure mode disabled."
        )

        self._notify(
            event
        )

        return self.get_status()

    # ============================================================
    # LEVEL MANAGEMENT
    # ============================================================

    def set_level(
        self,
        level: str,
        reason: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Change current security level.
        """

        return self.enable(
            level=level,
            reason=reason,
            force=force,
        )

    def _normalize_level(
        self,
        level: str,
    ) -> str:
        """
        Normalize security level.
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
                "Invalid secure mode level: %s",
                level,
            )

            return "restricted"

        return level

    # ============================================================
    # EXPIRATION
    # ============================================================

    def _check_expiration(
        self,
    ) -> bool:
        """
        Check whether temporary secure mode expired.
        """

        if not self.enabled:
            return False

        if self.expires_at is None:
            return False

        if datetime.now() < self.expires_at:
            return False

        previous_level = (
            self.current_level
        )

        self.enabled = False

        self.current_level = "normal"

        self.reason = (
            "Secure mode automatically expired."
        )

        self.enabled_at = None

        self.expires_at = None

        event = {
            "event": (
                "secure_mode_expired"
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

        logger.info(
            "Secure mode automatically expired."
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
    # POLICY CHECKING
    # ============================================================

    def get_policy(
        self,
        policy_name: str,
    ) -> Any:
        """
        Get current policy value.
        """

        with self._lock:

            self._check_expiration()

            policies = (
                self.policies.get(
                    self.current_level,
                    {},
                )
            )

            return policies.get(
                policy_name
            )

    def get_current_policies(
        self,
    ) -> Dict[str, Any]:
        """
        Return all policies for current mode.
        """

        with self._lock:

            self._check_expiration()

            return dict(
                self.policies.get(
                    self.current_level,
                    {},
                )
            )

    def is_operation_allowed(
        self,
        operation: str,
    ) -> bool:
        """
        Check whether an operation is allowed
        in the current security mode.
        """

        with self._lock:

            self._check_expiration()

            operation = (
                operation
                .lower()
                .strip()
            )

            policy_name = (
                self.OPERATION_POLICY_MAP.get(
                    operation
                )
            )

            if policy_name is None:

                return True

            policy_value = (
                self.get_policy(
                    policy_name
                )
            )

            return bool(
                policy_value
            )

    def requires_confirmation(
        self,
        operation: Optional[
            str
        ] = None,
    ) -> bool:
        """
        Check whether confirmation is required.

        Currently confirmation is controlled by
        the active secure mode policy.
        """

        with self._lock:

            self._check_expiration()

            return bool(
                self.get_policy(
                    "require_confirmation"
                )
            )

    # ============================================================
    # POLICY MANAGEMENT
    # ============================================================

    def update_policy(
        self,
        level: str,
        policy_name: str,
        value: Any,
    ) -> bool:
        """
        Update a policy value for a security level.
        """

        level = self._normalize_level(
            level
        )

        with self._lock:

            if level not in self.policies:

                return False

            self.policies[
                level
            ][policy_name] = value

            logger.info(
                "Policy updated | %s | %s",
                level,
                policy_name,
            )

            return True

    def reset_policies(
        self,
    ) -> None:
        """
        Reset all policies to defaults.
        """

        with self._lock:

            self.policies = {
                level: dict(policy)
                for level, policy in (
                    self.DEFAULT_POLICIES.items()
                )
            }

            self._load_custom_policies()

        logger.info(
            "Secure mode policies reset."
        )

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
        Register secure mode event handler.
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
        Remove an event handler.
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
        Notify registered handlers.
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
                    "Secure mode event handler "
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
        Add event to history.
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
        Return recent secure mode events.
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
        Clear secure mode history.
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
        Return current secure mode status.
        """

        with self._lock:

            self._check_expiration()

            return {
                "enabled": self.enabled,
                "level": (
                    self.current_level
                ),
                "reason": self.reason,
                "enabled_at": (
                    self.enabled_at.isoformat()
                    if self.enabled_at
                    else None
                ),
                "expires_at": (
                    self.expires_at.isoformat()
                    if self.expires_at
                    else None
                ),
                "policies": (
                    self.get_current_policies()
                ),
                "history_entries": (
                    len(self.history)
                ),
            }

    def is_enabled(
        self,
    ) -> bool:
        """
        Check whether secure mode is active.
        """

        with self._lock:

            self._check_expiration()

            return self.enabled

    def get_level(
        self,
    ) -> str:
        """
        Return current security level.
        """

        with self._lock:

            self._check_expiration()

            return self.current_level

    # ============================================================
    # EMERGENCY MODE
    # ============================================================

    def emergency_lockdown(
        self,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Immediately activate lockdown mode.
        """

        return self.enable(
            level="lockdown",
            reason=(
                reason
                or "Emergency security lockdown."
            ),
            force=True,
        )

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Shut down Secure Mode.
        """

        logger.info(
            "Shutting down SecureMode..."
        )

        with self._lock:

            self.event_handlers.clear()

        logger.info(
            "SecureMode shutdown complete."
        )
