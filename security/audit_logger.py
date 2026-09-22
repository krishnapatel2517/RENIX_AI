"""
RENIX Security Audit Logger
===========================

Provides structured security audit logging for RENIX.

Features:

* Security event logging
* Authentication event logging
* Permission audit logs
* Command execution logs
* Risk level tracking
* In-memory history
* JSON Lines persistent logs
* Event searching and filtering
* Automatic log rotation
  """

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("RENIX.AuditLogger")

class AuditLogger:
    """
    Structured audit logging system for RENIX.

    Example:

        audit = AuditLogger()

        audit.log(
            event_type="authentication",
            action="login",
            user_id="krishna",
            status="success",
        )
    """

    VALID_SEVERITIES = {
        "info",
        "low",
        "medium",
        "high",
        "critical",
    }

    VALID_STATUSES = {
        "success",
        "failure",
        "blocked",
        "warning",
        "pending",
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize audit logger.
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

        self.persist_logs = bool(
            self.config.get(
                "persist_logs",
                True,
            )
        )

        self.log_directory = Path(
            self.config.get(
                "log_directory",
                "logs/audit",
            )
        )

        self.max_file_size = int(
            self.config.get(
                "max_file_size",
                10 * 1024 * 1024,
            )
        )

        self.history: List[
            Dict[str, Any]
        ] = []

        self._lock = threading.Lock()

        if self.persist_logs:

            self.log_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        logger.info(
            "AuditLogger initialized."
        )

    # ============================================================
    # FILE PATH
    # ============================================================

    def _get_log_file(
        self,
    ) -> Path:
        """
        Return today's audit log file.
        """

        date_string = (
            datetime.now().strftime(
                "%Y-%m-%d"
            )
        )

        return (
            self.log_directory
            / f"audit_{date_string}.jsonl"
        )

    # ============================================================
    # ROTATION
    # ============================================================

    def _rotate_if_needed(
        self,
        log_file: Path,
    ) -> None:
        """
        Rotate log file if maximum size
        has been exceeded.
        """

        if not log_file.exists():
            return

        try:

            if (
                log_file.stat().st_size
                < self.max_file_size
            ):

                return

            timestamp = (
                datetime.now().strftime(
                    "%H%M%S"
                )
            )

            rotated_file = (
                log_file.with_name(
                    f"{log_file.stem}"
                    f"_{timestamp}"
                    f"{log_file.suffix}"
                )
            )

            log_file.rename(
                rotated_file
            )

            logger.info(
                "Audit log rotated: %s",
                rotated_file,
            )

        except Exception as error:

            logger.error(
                "Failed to rotate audit log: %s",
                error,
            )

    # ============================================================
    # NORMALIZATION
    # ============================================================

    def _normalize_severity(
        self,
        severity: str,
    ) -> str:
        """
        Normalize severity value.
        """

        severity = (
            severity
            or "info"
        ).lower().strip()

        if severity not in (
            self.VALID_SEVERITIES
        ):

            return "info"

        return severity

    def _normalize_status(
        self,
        status: str,
    ) -> str:
        """
        Normalize event status.
        """

        status = (
            status
            or "success"
        ).lower().strip()

        if status not in (
            self.VALID_STATUSES
        ):

            return "success"

        return status

    # ============================================================
    # CREATE EVENT
    # ============================================================

    def create_event(
        self,
        event_type: str,
        action: str,
        user_id: Optional[str] = None,
        status: str = "success",
        severity: str = "info",
        details: Optional[
            Dict[str, Any]
        ] = None,
        source: str = "RENIX",
    ) -> Dict[str, Any]:
        """
        Create a structured audit event.
        """

        event = {
            "timestamp": (
                datetime.now().isoformat()
            ),
            "event_type": (
                event_type.strip().lower()
            ),
            "action": (
                action.strip()
            ),
            "user_id": user_id,
            "status": (
                self._normalize_status(
                    status
                )
            ),
            "severity": (
                self._normalize_severity(
                    severity
                )
            ),
            "source": source,
            "details": details or {},
        }

        return event

    # ============================================================
    # LOG EVENT
    # ============================================================

    def log(
        self,
        event_type: str,
        action: str,
        user_id: Optional[str] = None,
        status: str = "success",
        severity: str = "info",
        details: Optional[
            Dict[str, Any]
        ] = None,
        source: str = "RENIX",
    ) -> Dict[str, Any]:
        """
        Create and store an audit event.
        """

        event = self.create_event(
            event_type=event_type,
            action=action,
            user_id=user_id,
            status=status,
            severity=severity,
            details=details,
            source=source,
        )

        if not self.enabled:

            return event

        with self._lock:

            self.history.append(
                event
            )

            self._trim_history()

            if self.persist_logs:

                self._write_event(
                    event
                )

        logger.info(
            "Audit event | %s | %s | %s",
            event_type,
            action,
            status,
        )

        return event.copy()

    # ============================================================
    # WRITE EVENT
    # ============================================================

    def _write_event(
        self,
        event: Dict[str, Any],
    ) -> bool:
        """
        Write event to JSON Lines audit file.
        """

        try:

            log_file = (
                self._get_log_file()
            )

            self._rotate_if_needed(
                log_file
            )

            with log_file.open(
                "a",
                encoding="utf-8",
            ) as file:

                json.dump(
                    event,
                    file,
                    ensure_ascii=False,
                    default=str,
                )

                file.write(
                    "\n"
                )

            return True

        except Exception as error:

            logger.error(
                "Failed to write audit event: %s",
                error,
            )

            return False

    # ============================================================
    # HISTORY MANAGEMENT
    # ============================================================

    def _trim_history(
        self,
    ) -> None:
        """
        Keep history within configured size.
        """

        if (
            len(self.history)
            <= self.max_history
        ):

            return

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
    ) -> List[Dict[str, Any]]:
        """
        Return recent audit events.
        """

        limit = max(
            1,
            limit,
        )

        with self._lock:

            return [
                event.copy()
                for event in self.history[
                    -limit:
                ]
            ]

    def clear_history(
        self,
    ) -> None:
        """
        Clear in-memory audit history.
        """

        with self._lock:

            self.history.clear()

    # ============================================================
    # SEARCH EVENTS
    # ============================================================

    def search(
        self,
        event_type: Optional[
            str
        ] = None,
        user_id: Optional[
            str
        ] = None,
        status: Optional[
            str
        ] = None,
        severity: Optional[
            str
        ] = None,
        action_contains: Optional[
            str
        ] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search in-memory audit history.
        """

        results = []

        with self._lock:

            events = list(
                self.history
            )

        for event in reversed(
            events
        ):

            if (
                event_type
                and event.get(
                    "event_type"
                ) != event_type.lower()
            ):
                continue

            if (
                user_id
                and event.get(
                    "user_id"
                ) != user_id
            ):
                continue

            if (
                status
                and event.get(
                    "status"
                ) != status.lower()
            ):
                continue

            if (
                severity
                and event.get(
                    "severity"
                ) != severity.lower()
            ):
                continue

            if action_contains:

                action = (
                    event.get(
                        "action",
                        ""
                    ).lower()
                )

                if (
                    action_contains.lower()
                    not in action
                ):
                    continue

            results.append(
                event.copy()
            )

            if len(results) >= limit:
                break

        return results

    # ============================================================
    # SPECIALIZED LOGGING
    # ============================================================

    def log_authentication(
        self,
        user_id: Optional[str],
        action: str,
        success: bool,
        details: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Log authentication event.
        """

        return self.log(
            event_type="authentication",
            action=action,
            user_id=user_id,
            status=(
                "success"
                if success
                else "failure"
            ),
            severity=(
                "info"
                if success
                else "high"
            ),
            details=details,
        )

    def log_permission(
        self,
        user_id: Optional[str],
        action: str,
        allowed: bool,
        details: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Log permission decision.
        """

        return self.log(
            event_type="permission",
            action=action,
            user_id=user_id,
            status=(
                "success"
                if allowed
                else "blocked"
            ),
            severity=(
                "info"
                if allowed
                else "medium"
            ),
            details=details,
        )

    def log_command(
        self,
        user_id: Optional[str],
        command: str,
        status: str,
        risk_level: str = "info",
        details: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Log command execution event.
        """

        return self.log(
            event_type="command",
            action=command,
            user_id=user_id,
            status=status,
            severity=risk_level,
            details=details,
        )

    def log_security_event(
        self,
        action: str,
        severity: str = "medium",
        user_id: Optional[
            str
        ] = None,
        details: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Log general security event.
        """

        return self.log(
            event_type="security",
            action=action,
            user_id=user_id,
            status="warning",
            severity=severity,
            details=details,
        )

    # ============================================================
    # STATISTICS
    # ============================================================

    def get_statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Return audit event statistics.
        """

        with self._lock:

            events = list(
                self.history
            )

        statistics = {
            "total_events": len(
                events
            ),
            "by_status": {},
            "by_severity": {},
            "by_type": {},
        }

        for event in events:

            status = event.get(
                "status",
                "unknown",
            )

            severity = event.get(
                "severity",
                "unknown",
            )

            event_type = event.get(
                "event_type",
                "unknown",
            )

            statistics[
                "by_status"
            ][status] = (
                statistics[
                    "by_status"
                ].get(
                    status,
                    0,
                )
                + 1
            )

            statistics[
                "by_severity"
            ][severity] = (
                statistics[
                    "by_severity"
                ].get(
                    severity,
                    0,
                )
                + 1
            )

            statistics[
                "by_type"
            ][event_type] = (
                statistics[
                    "by_type"
                ].get(
                    event_type,
                    0,
                )
                + 1
            )

        return statistics

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return audit logger status.
        """

        return {
            "enabled": self.enabled,
            "persist_logs": (
                self.persist_logs
            ),
            "log_directory": str(
                self.log_directory
            ),
            "history_entries": len(
                self.history
            ),
            "max_history": (
                self.max_history
            ),
            "max_file_size": (
                self.max_file_size
            ),
        }

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Safely shut down audit logger.
        """

        logger.info(
            "Shutting down AuditLogger..."
        )

        logger.info(
            "AuditLogger shutdown complete."
        )
