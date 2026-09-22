"""
RENIX Threat Monitor
====================

Monitors security-related events and detects suspicious patterns.

Features:

* Suspicious activity detection
* Failed authentication tracking
* Rate-based threat detection
* Threat severity scoring
* Temporary user blocking
* Security alerts
* Threat history
* Integration-ready event monitoring

Note:
This module provides application-level threat monitoring.
For production-grade protection, it should be combined with
OS security, firewalling, endpoint protection, and proper
authentication infrastructure.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("RENIX.ThreatMonitor")

class ThreatMonitor:
    """
    Monitors RENIX for suspicious activity and threats.

    Example:

        monitor = ThreatMonitor()

        result = monitor.record_event(
            event_type="authentication_failure",
            user_id="unknown_user"
        )

        if result["threat_detected"]:
            print("Security alert!")
    """

    SEVERITY_LEVELS = {
        "info": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize threat monitor.
        """

        self.config = config or {}

        self.enabled = bool(
            self.config.get(
                "enabled",
                True,
            )
        )

        self.max_history = int(
            self.config.get(
                "max_history",
                5000,
            )
        )

        self.failed_auth_threshold = int(
            self.config.get(
                "failed_auth_threshold",
                5,
            )
        )

        self.failed_auth_window = int(
            self.config.get(
                "failed_auth_window",
                300,
            )
        )

        self.block_duration = int(
            self.config.get(
                "block_duration",
                900,
            )
        )

        self.suspicious_command_threshold = int(
            self.config.get(
                "suspicious_command_threshold",
                5,
            )
        )

        self.suspicious_command_window = int(
            self.config.get(
                "suspicious_command_window",
                300,
            )
        )

        self.events: List[
            Dict[str, Any]
        ] = []

        self.threats: List[
            Dict[str, Any]
        ] = []

        self.failed_auth_attempts = defaultdict(
            list
        )

        self.suspicious_commands = defaultdict(
            list
        )

        self.blocked_entities: Dict[
            str,
            Dict[str, Any]
        ] = {}

        self.alert_handlers = []

        self._lock = threading.RLock()

        logger.info(
            "ThreatMonitor initialized."
        )

    # ============================================================
    # EVENT RECORDING
    # ============================================================

    def record_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        source: str = "RENIX",
        severity: str = "info",
        details: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Record and analyze a security event.
        """

        event = {
            "timestamp": (
                datetime.now().isoformat()
            ),
            "event_type": (
                event_type.lower().strip()
            ),
            "user_id": user_id,
            "source": source,
            "severity": (
                self._normalize_severity(
                    severity
                )
            ),
            "details": details or {},
        }

        result = {
            "event": event,
            "threat_detected": False,
            "threats": [],
            "blocked": False,
        }

        if not self.enabled:
            return result

        with self._lock:

            self.events.append(
                event
            )

            self._trim_history()

            detected_threats = (
                self._analyze_event(
                    event
                )
            )

            if detected_threats:

                result[
                    "threat_detected"
                ] = True

                result[
                    "threats"
                ] = detected_threats

            if user_id:

                result["blocked"] = (
                    self.is_blocked(
                        user_id
                    )
                )

        return result

    # ============================================================
    # EVENT ANALYSIS
    # ============================================================

    def _analyze_event(
        self,
        event: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Analyze security event for threats.
        """

        threats = []

        event_type = event[
            "event_type"
        ]

        user_id = event.get(
            "user_id"
        )

        if (
            event_type
            in {
                "authentication_failure",
                "login_failure",
                "voice_auth_failure",
                "face_auth_failure",
            }
        ):

            threat = (
                self._track_failed_authentication(
                    user_id
                )
            )

            if threat:

                threats.append(
                    threat
                )

        if (
            event_type
            in {
                "dangerous_command",
                "blocked_command",
                "suspicious_command",
            }
        ):

            threat = (
                self._track_suspicious_command(
                    user_id
                )
            )

            if threat:

                threats.append(
                    threat
                )

        if (
            event_type
            == "permission_violation"
        ):

            threat = self._create_threat(
                threat_type=(
                    "permission_violation"
                ),
                severity="medium",
                user_id=user_id,
                description=(
                    "Unauthorized permission "
                    "access attempt detected."
                ),
                details=event.get(
                    "details",
                    {}
                ),
            )

            threats.append(
                threat
            )

        if (
            event_type
            == "security_bypass_attempt"
        ):

            threat = self._create_threat(
                threat_type=(
                    "security_bypass_attempt"
                ),
                severity="critical",
                user_id=user_id,
                description=(
                    "Attempt to bypass RENIX "
                    "security controls detected."
                ),
                details=event.get(
                    "details",
                    {}
                ),
            )

            if user_id:

                self.block_entity(
                    user_id,
                    reason=(
                        "Security bypass attempt."
                    ),
                    severity="critical",
                )

            threats.append(
                threat
            )

        return threats

    # ============================================================
    # FAILED AUTHENTICATION TRACKING
    # ============================================================

    def _track_failed_authentication(
        self,
        user_id: Optional[str],
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Track repeated authentication failures.
        """

        if not user_id:
            user_id = "unknown"

        now = datetime.now()

        attempts = (
            self.failed_auth_attempts[
                user_id
            ]
        )

        attempts.append(
            now
        )

        window_start = (
            now
            - timedelta(
                seconds=self.failed_auth_window
            )
        )

        self.failed_auth_attempts[
            user_id
        ] = [
            timestamp
            for timestamp in attempts
            if timestamp >= window_start
        ]

        count = len(
            self.failed_auth_attempts[
                user_id
            ]
        )

        if (
            count
            < self.failed_auth_threshold
        ):

            return None

        threat = self._create_threat(
            threat_type=(
                "brute_force_attempt"
            ),
            severity="high",
            user_id=user_id,
            description=(
                f"{count} failed authentication "
                f"attempts detected."
            ),
            details={
                "attempt_count": count,
                "window_seconds": (
                    self.failed_auth_window
                ),
            },
        )

        self.block_entity(
            user_id,
            reason=(
                "Too many failed authentication "
                "attempts."
            ),
            severity="high",
        )

        return threat

    # ============================================================
    # SUSPICIOUS COMMAND TRACKING
    # ============================================================

    def _track_suspicious_command(
        self,
        user_id: Optional[str],
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Detect repeated suspicious commands.
        """

        if not user_id:
            user_id = "unknown"

        now = datetime.now()

        commands = (
            self.suspicious_commands[
                user_id
            ]
        )

        commands.append(
            now
        )

        window_start = (
            now
            - timedelta(
                seconds=(
                    self.suspicious_command_window
                )
            )
        )

        self.suspicious_commands[
            user_id
        ] = [
            timestamp
            for timestamp in commands
            if timestamp >= window_start
        ]

        count = len(
            self.suspicious_commands[
                user_id
            ]
        )

        if (
            count
            < self.suspicious_command_threshold
        ):

            return None

        threat = self._create_threat(
            threat_type=(
                "repeated_suspicious_commands"
            ),
            severity="high",
            user_id=user_id,
            description=(
                f"{count} suspicious commands "
                f"detected."
            ),
            details={
                "command_count": count,
                "window_seconds": (
                    self.suspicious_command_window
                ),
            },
        )

        return threat

    # ============================================================
    # THREAT CREATION
    # ============================================================

    def _create_threat(
        self,
        threat_type: str,
        severity: str,
        user_id: Optional[str],
        description: str,
        details: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Create and store a threat record.
        """

        threat = {
            "timestamp": (
                datetime.now().isoformat()
            ),
            "threat_type": threat_type,
            "severity": (
                self._normalize_severity(
                    severity
                )
            ),
            "user_id": user_id,
            "description": description,
            "details": details or {},
            "resolved": False,
        }

        self.threats.append(
            threat
        )

        self._trim_threats()

        logger.warning(
            "Threat detected | %s | %s",
            threat_type,
            severity,
        )

        self._send_alert(
            threat
        )

        return threat.copy()

    # ============================================================
    # BLOCKING
    # ============================================================

    def block_entity(
        self,
        entity_id: str,
        reason: str,
        severity: str = "high",
        duration: Optional[
            int
        ] = None,
    ) -> Dict[str, Any]:
        """
        Temporarily block an entity.
        """

        block_seconds = (
            duration
            if duration is not None
            else self.block_duration
        )

        expires_at = (
            datetime.now()
            + timedelta(
                seconds=block_seconds
            )
        )

        block = {
            "entity_id": entity_id,
            "reason": reason,
            "severity": (
                self._normalize_severity(
                    severity
                )
            ),
            "blocked_at": (
                datetime.now().isoformat()
            ),
            "expires_at": (
                expires_at.isoformat()
            ),
        }

        self.blocked_entities[
            entity_id
        ] = block

        logger.warning(
            "Entity blocked: %s",
            entity_id,
        )

        return block.copy()

    def unblock_entity(
        self,
        entity_id: str,
    ) -> bool:
        """
        Remove entity from block list.
        """

        if entity_id not in (
            self.blocked_entities
        ):
            return False

        del self.blocked_entities[
            entity_id
        ]

        logger.info(
            "Entity unblocked: %s",
            entity_id,
        )

        return True

    def is_blocked(
        self,
        entity_id: str,
    ) -> bool:
        """
        Check whether entity is blocked.
        """

        block = (
            self.blocked_entities.get(
                entity_id
            )
        )

        if block is None:
            return False

        expires_at = (
            datetime.fromisoformat(
                block["expires_at"]
            )
        )

        if datetime.now() >= expires_at:

            self.unblock_entity(
                entity_id
            )

            return False

        return True

    def get_block(
        self,
        entity_id: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Get active block information.
        """

        if not self.is_blocked(
            entity_id
        ):

            return None

        block = (
            self.blocked_entities.get(
                entity_id
            )
        )

        return (
            block.copy()
            if block
            else None
        )

    # ============================================================
    # ALERT HANDLERS
    # ============================================================

    def add_alert_handler(
        self,
        handler,
    ) -> None:
        """
        Register a threat alert handler.

        Handler receives a threat dictionary.
        """

        if callable(handler):

            self.alert_handlers.append(
                handler
            )

    def remove_alert_handler(
        self,
        handler,
    ) -> bool:
        """
        Remove registered alert handler.
        """

        if handler not in (
            self.alert_handlers
        ):

            return False

        self.alert_handlers.remove(
            handler
        )

        return True

    def _send_alert(
        self,
        threat: Dict[str, Any],
    ) -> None:
        """
        Send threat to registered handlers.
        """

        for handler in list(
            self.alert_handlers
        ):

            try:

                handler(
                    threat.copy()
                )

            except Exception as error:

                logger.error(
                    "Threat alert handler failed: %s",
                    error,
                )

    # ============================================================
    # RESOLVE THREATS
    # ============================================================

    def resolve_threat(
        self,
        index: int,
        resolution: Optional[
            str
        ] = None,
    ) -> bool:
        """
        Mark a threat as resolved.
        """

        if (
            index < 0
            or index >= len(
                self.threats
            )
        ):

            return False

        threat = self.threats[
            index
        ]

        threat["resolved"] = True

        threat["resolved_at"] = (
            datetime.now().isoformat()
        )

        threat["resolution"] = (
            resolution
            or "Resolved manually."
        )

        return True

    # ============================================================
    # CLEANUP
    # ============================================================

    def cleanup_expired_blocks(
        self,
    ) -> int:
        """
        Remove expired temporary blocks.
        """

        expired = []

        for (
            entity_id,
            block,
        ) in list(
            self.blocked_entities.items()
        ):

            expires_at = (
                datetime.fromisoformat(
                    block["expires_at"]
                )
            )

            if (
                datetime.now()
                >= expires_at
            ):

                expired.append(
                    entity_id
                )

        for entity_id in expired:

            self.unblock_entity(
                entity_id
            )

        return len(
            expired
        )

    def _trim_history(
        self,
    ) -> None:
        """
        Limit event history size.
        """

        if len(self.events) > (
            self.max_history
        ):

            overflow = (
                len(self.events)
                - self.max_history
            )

            self.events = (
                self.events[
                    overflow:
                ]
            )

    def _trim_threats(
        self,
    ) -> None:
        """
        Limit threat history size.
        """

        if len(self.threats) > (
            self.max_history
        ):

            overflow = (
                len(self.threats)
                - self.max_history
            )

            self.threats = (
                self.threats[
                    overflow:
                ]
            )

    # ============================================================
    # HISTORY
    # ============================================================

    def get_events(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Return recent security events.
        """

        return [
            event.copy()
            for event in self.events[
                -max(1, limit):
            ]
        ]

    def get_threats(
        self,
        unresolved_only: bool = False,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Return detected threats.
        """

        threats = self.threats

        if unresolved_only:

            threats = [
                threat
                for threat in threats
                if not threat.get(
                    "resolved",
                    False,
                )
            ]

        return [
            threat.copy()
            for threat in threats[
                -max(1, limit):
            ]
        ]

    # ============================================================
    # SEVERITY
    # ============================================================

    def _normalize_severity(
        self,
        severity: str,
    ) -> str:
        """
        Normalize threat severity.
        """

        severity = (
            severity
            or "info"
        ).lower().strip()

        if severity not in (
            self.SEVERITY_LEVELS
        ):

            return "info"

        return severity

    # ============================================================
    # STATISTICS
    # ============================================================

    def get_statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Return threat monitoring statistics.
        """

        unresolved = sum(
            1
            for threat in self.threats
            if not threat.get(
                "resolved",
                False,
            )
        )

        severity_counts = {}

        for threat in self.threats:

            severity = threat.get(
                "severity",
                "info",
            )

            severity_counts[
                severity
            ] = (
                severity_counts.get(
                    severity,
                    0,
                )
                + 1
            )

        return {
            "total_events": len(
                self.events
            ),
            "total_threats": len(
                self.threats
            ),
            "unresolved_threats": (
                unresolved
            ),
            "blocked_entities": len(
                self.blocked_entities
            ),
            "threats_by_severity": (
                severity_counts
            ),
        }

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return threat monitor status.
        """

        self.cleanup_expired_blocks()

        return {
            "enabled": self.enabled,
            "events": len(
                self.events
            ),
            "threats": len(
                self.threats
            ),
            "blocked_entities": len(
                self.blocked_entities
            ),
            "failed_auth_threshold": (
                self.failed_auth_threshold
            ),
            "suspicious_command_threshold": (
                self.suspicious_command_threshold
            ),
        }

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Safely shut down threat monitor.
        """

        logger.info(
            "Shutting down ThreatMonitor..."
        )

        self.cleanup_expired_blocks()

        self.alert_handlers.clear()

        logger.info(
            "ThreatMonitor shutdown complete."
        )
