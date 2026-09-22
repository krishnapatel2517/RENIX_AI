"""
RENIX Security Confirmation System
==================================

Handles user confirmation for sensitive or dangerous actions.

Features:

* Confirmation requests
* Risk levels
* Action approval/rejection
* Confirmation timeouts
* One-time confirmations
* Batch confirmations
* Pending confirmation tracking
* Security audit information
  """

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("RENIX.Confirmation")

class ConfirmationManager:
    """
    Manages security confirmations for RENIX.

    Example:

        confirmation = manager.request_confirmation(
            user_id="krishna",
            action="Delete all files",
            risk_level="high",
        )

        manager.approve(
            confirmation_id
        )
    """

    RISK_LEVELS = {
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
        Initialize confirmation manager.
        """

        self.config = config or {}

        self.enabled = self.config.get(
            "enabled",
            True,
        )

        self.default_timeout = int(
            self.config.get(
                "default_timeout",
                60,
            )
        )

        self.auto_confirm_low_risk = bool(
            self.config.get(
                "auto_confirm_low_risk",
                False,
            )
        )

        self.confirmations: Dict[
            str,
            Dict[str, Any]
        ] = {}

        self.history: List[
            Dict[str, Any]
        ] = []

        self.max_history = int(
            self.config.get(
                "max_history",
                1000,
            )
        )

        logger.info(
            "ConfirmationManager initialized."
        )

    # ============================================================
    # CREATE CONFIRMATION
    # ============================================================

    def request_confirmation(
        self,
        user_id: str,
        action: str,
        risk_level: str = "medium",
        message: Optional[str] = None,
        timeout: Optional[int] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Create a confirmation request.

        Args:
            user_id:
                User required to approve the action.

            action:
                Action requiring confirmation.

            risk_level:
                low, medium, high, or critical.

            message:
                Custom confirmation message.

            timeout:
                Time before confirmation expires.

            metadata:
                Additional action data.
        """

        if risk_level not in self.RISK_LEVELS:
            risk_level = "medium"

        confirmation_id = str(
            uuid.uuid4()
        )

        timeout_seconds = (
            timeout
            if timeout is not None
            else self.default_timeout
        )

        created_at = datetime.now()

        expires_at = (
            created_at
            + timedelta(
                seconds=timeout_seconds
            )
        )

        # Low-risk actions may be configured
        # for automatic approval.
        auto_approved = (
            self.auto_confirm_low_risk
            and risk_level == "low"
        )

        confirmation = {
            "confirmation_id": (
                confirmation_id
            ),
            "user_id": user_id,
            "action": action,
            "risk_level": risk_level,
            "message": (
                message
                or self._build_message(
                    action,
                    risk_level,
                )
            ),
            "metadata": metadata or {},
            "status": (
                "approved"
                if auto_approved
                else "pending"
            ),
            "created_at": (
                created_at.isoformat()
            ),
            "expires_at": (
                expires_at.isoformat()
            ),
            "timeout_seconds": (
                timeout_seconds
            ),
            "resolved_at": (
                created_at.isoformat()
                if auto_approved
                else None
            ),
        }

        self.confirmations[
            confirmation_id
        ] = confirmation

        logger.info(
            "Confirmation requested: %s | %s",
            confirmation_id,
            action,
        )

        if auto_approved:

            self._move_to_history(
                confirmation_id
            )

        return confirmation.copy()

    # ============================================================
    # BUILD CONFIRMATION MESSAGE
    # ============================================================

    @staticmethod
    def _build_message(
        action: str,
        risk_level: str,
    ) -> str:
        """
        Create a standard confirmation message.
        """

        prefix = {
            "low": (
                "Please confirm this action"
            ),
            "medium": (
                "This action requires confirmation"
            ),
            "high": (
                "Warning: This action may have "
                "important consequences"
            ),
            "critical": (
                "CRITICAL ACTION: Please carefully "
                "review before approving"
            ),
        }.get(
            risk_level,
            "Confirmation required",
        )

        return (
            f"{prefix}: {action}"
        )

    # ============================================================
    # GET CONFIRMATION
    # ============================================================

    def get_confirmation(
        self,
        confirmation_id: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Get confirmation information.
        """

        self.cleanup_expired()

        confirmation = self.confirmations.get(
            confirmation_id
        )

        if confirmation is None:
            return None

        return confirmation.copy()

    # ============================================================
    # CHECK EXPIRATION
    # ============================================================

    def _is_expired(
        self,
        confirmation: Dict[str, Any],
    ) -> bool:
        """
        Check whether a confirmation expired.
        """

        if confirmation.get(
            "status"
        ) != "pending":

            return False

        expires_at = datetime.fromisoformat(
            confirmation[
                "expires_at"
            ]
        )

        return (
            datetime.now()
            >= expires_at
        )

    # ============================================================
    # APPROVE CONFIRMATION
    # ============================================================

    def approve(
        self,
        confirmation_id: str,
        user_id: Optional[
            str
        ] = None,
    ) -> Dict[str, Any]:
        """
        Approve a pending confirmation.
        """

        result = {
            "success": False,
            "confirmation_id": (
                confirmation_id
            ),
            "status": None,
            "reason": "",
        }

        confirmation = self.confirmations.get(
            confirmation_id
        )

        if confirmation is None:

            result["reason"] = (
                "Confirmation not found."
            )

            return result

        if (
            user_id is not None
            and confirmation["user_id"]
            != user_id
        ):

            result["reason"] = (
                "User is not authorized to "
                "approve this confirmation."
            )

            return result

        if self._is_expired(
            confirmation
        ):

            confirmation[
                "status"
            ] = "expired"

            confirmation[
                "resolved_at"
            ] = (
                datetime.now().isoformat()
            )

            result["status"] = "expired"

            result["reason"] = (
                "Confirmation has expired."
            )

            self._move_to_history(
                confirmation_id
            )

            return result

        if confirmation[
            "status"
        ] != "pending":

            result["status"] = (
                confirmation["status"]
            )

            result["reason"] = (
                "Confirmation is no longer pending."
            )

            return result

        confirmation[
            "status"
        ] = "approved"

        confirmation[
            "resolved_at"
        ] = datetime.now().isoformat()

        result["success"] = True

        result["status"] = "approved"

        result["reason"] = (
            "Action approved."
        )

        logger.info(
            "Confirmation approved: %s",
            confirmation_id,
        )

        self._move_to_history(
            confirmation_id
        )

        return result

    # ============================================================
    # REJECT CONFIRMATION
    # ============================================================

    def reject(
        self,
        confirmation_id: str,
        user_id: Optional[
            str
        ] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Reject a pending confirmation.
        """

        result = {
            "success": False,
            "confirmation_id": (
                confirmation_id
            ),
            "status": None,
            "reason": "",
        }

        confirmation = self.confirmations.get(
            confirmation_id
        )

        if confirmation is None:

            result["reason"] = (
                "Confirmation not found."
            )

            return result

        if (
            user_id is not None
            and confirmation["user_id"]
            != user_id
        ):

            result["reason"] = (
                "User is not authorized to "
                "reject this confirmation."
            )

            return result

        if confirmation[
            "status"
        ] != "pending":

            result["status"] = (
                confirmation["status"]
            )

            result["reason"] = (
                "Confirmation is no longer pending."
            )

            return result

        confirmation[
            "status"
        ] = "rejected"

        confirmation[
            "resolved_at"
        ] = datetime.now().isoformat()

        confirmation[
            "rejection_reason"
        ] = (
            reason
            or "Rejected by user."
        )

        result["success"] = True

        result["status"] = "rejected"

        result["reason"] = (
            confirmation[
                "rejection_reason"
            ]
        )

        logger.info(
            "Confirmation rejected: %s",
            confirmation_id,
        )

        self._move_to_history(
            confirmation_id
        )

        return result

    # ============================================================
    # CANCEL CONFIRMATION
    # ============================================================

    def cancel(
        self,
        confirmation_id: str,
        reason: str = "Cancelled.",
    ) -> bool:
        """
        Cancel a pending confirmation.
        """

        confirmation = self.confirmations.get(
            confirmation_id
        )

        if confirmation is None:
            return False

        if confirmation[
            "status"
        ] != "pending":

            return False

        confirmation[
            "status"
        ] = "cancelled"

        confirmation[
            "resolved_at"
        ] = datetime.now().isoformat()

        confirmation[
            "cancellation_reason"
        ] = reason

        self._move_to_history(
            confirmation_id
        )

        logger.info(
            "Confirmation cancelled: %s",
            confirmation_id,
        )

        return True

    # ============================================================
    # CHECK STATUS
    # ============================================================

    def is_approved(
        self,
        confirmation_id: str,
    ) -> bool:
        """
        Check whether confirmation is approved.
        """

        confirmation = self.get_confirmation(
            confirmation_id
        )

        if confirmation is None:
            return False

        return (
            confirmation[
                "status"
            ] == "approved"
        )

    def is_pending(
        self,
        confirmation_id: str,
    ) -> bool:
        """
        Check whether confirmation is pending.
        """

        confirmation = self.confirmations.get(
            confirmation_id
        )

        if confirmation is None:
            return False

        if self._is_expired(
            confirmation
        ):

            self.cleanup_expired()

            return False

        return (
            confirmation[
                "status"
            ] == "pending"
        )

    # ============================================================
    # PENDING CONFIRMATIONS
    # ============================================================

    def get_pending(
        self,
        user_id: Optional[
            str
        ] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return all pending confirmations.
        """

        self.cleanup_expired()

        confirmations = []

        for confirmation in (
            self.confirmations.values()
        ):

            if (
                confirmation["status"]
                != "pending"
            ):
                continue

            if (
                user_id is not None
                and confirmation["user_id"]
                != user_id
            ):
                continue

            confirmations.append(
                confirmation.copy()
            )

        return confirmations

    # ============================================================
    # CLEANUP EXPIRED
    # ============================================================

    def cleanup_expired(
        self,
    ) -> int:
        """
        Mark expired confirmations and move
        them into history.
        """

        expired_ids = []

        for (
            confirmation_id,
            confirmation,
        ) in list(
            self.confirmations.items()
        ):

            if self._is_expired(
                confirmation
            ):

                confirmation[
                    "status"
                ] = "expired"

                confirmation[
                    "resolved_at"
                ] = (
                    datetime.now().isoformat()
                )

                expired_ids.append(
                    confirmation_id
                )

        for confirmation_id in expired_ids:

            self._move_to_history(
                confirmation_id
            )

        if expired_ids:

            logger.info(
                "%s expired confirmations cleaned.",
                len(expired_ids),
            )

        return len(expired_ids)

    # ============================================================
    # MOVE TO HISTORY
    # ============================================================

    def _move_to_history(
        self,
        confirmation_id: str,
    ) -> None:
        """
        Move a completed confirmation
        to history.
        """

        confirmation = self.confirmations.pop(
            confirmation_id,
            None,
        )

        if confirmation is None:
            return

        self.history.append(
            confirmation.copy()
        )

        if len(self.history) > self.max_history:

            overflow = (
                len(self.history)
                - self.max_history
            )

            self.history = self.history[
                overflow:
            ]

    # ============================================================
    # HISTORY
    # ============================================================

    def get_history(
        self,
        user_id: Optional[
            str
        ] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Return confirmation history.
        """

        history = self.history

        if user_id:

            history = [
                item
                for item in history
                if item.get(
                    "user_id"
                ) == user_id
            ]

        return history[
            -max(1, limit):
        ]

    def clear_history(
        self,
    ) -> None:
        """
        Clear confirmation history.
        """

        self.history.clear()

    # ============================================================
    # BATCH CONFIRMATIONS
    # ============================================================

    def request_batch_confirmation(
        self,
        user_id: str,
        actions: List[str],
        risk_level: str = "high",
        timeout: Optional[int] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Request confirmation for multiple actions.
        """

        action_text = "; ".join(
            actions
        )

        return self.request_confirmation(
            user_id=user_id,
            action=(
                f"Batch operation: {action_text}"
            ),
            risk_level=risk_level,
            timeout=timeout,
            metadata={
                "actions": actions,
                **(metadata or {}),
            },
        )

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return confirmation system status.
        """

        self.cleanup_expired()

        pending = sum(
            1
            for confirmation
            in self.confirmations.values()
            if confirmation[
                "status"
            ] == "pending"
        )

        return {
            "enabled": self.enabled,
            "pending_confirmations": pending,
            "history_entries": len(
                self.history
            ),
            "default_timeout": (
                self.default_timeout
            ),
            "auto_confirm_low_risk": (
                self.auto_confirm_low_risk
            ),
        }

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Safely shut down confirmation manager.
        """

        logger.info(
            "Shutting down ConfirmationManager..."
        )

        self.cleanup_expired()

        self.confirmations.clear()

        logger.info(
            "ConfirmationManager shutdown complete."
        )
