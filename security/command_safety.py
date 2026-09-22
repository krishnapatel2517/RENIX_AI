"""
RENIX Command Safety System
===========================

Analyzes commands before execution and determines
whether they are safe, suspicious, dangerous, or
require user confirmation.

Features:

* Command risk analysis
* Dangerous command detection
* Sensitive operation detection
* Confirmation requirements
* Custom safety rules
* Risk scoring
* Command history
* Blocked command patterns
  """

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("RENIX.CommandSafety")

class CommandSafety:
    """
    Security layer that evaluates commands before
    RENIX executes them.

    Example:

        result = safety.analyze(
            command="delete all files"
        )

        if result["allowed"]:
            execute_command()
    """

    RISK_LEVELS = {
        "safe": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }

    DEFAULT_DANGEROUS_PATTERNS = [
        {
            "pattern": r"\brm\s+-rf\s+/",
            "risk": "critical",
            "reason": (
                "Potential destructive root "
                "filesystem deletion."
            ),
            "blocked": True,
        },
        {
            "pattern": r"\bformat\b",
            "risk": "critical",
            "reason": (
                "Potential disk formatting "
                "operation."
            ),
            "blocked": False,
        },
        {
            "pattern": r"\bshutdown\b",
            "risk": "high",
            "reason": (
                "System shutdown operation."
            ),
            "blocked": False,
        },
        {
            "pattern": r"\brestart\b",
            "risk": "medium",
            "reason": (
                "System restart operation."
            ),
            "blocked": False,
        },
        {
            "pattern": r"\bdelete\b.*\ball\b",
            "risk": "critical",
            "reason": (
                "Mass deletion operation detected."
            ),
            "blocked": False,
        },
        {
            "pattern": r"\bdelete\b.*\bfiles?\b",
            "risk": "high",
            "reason": (
                "File deletion operation detected."
            ),
            "blocked": False,
        },
        {
            "pattern": r"\brmdir\b",
            "risk": "high",
            "reason": (
                "Directory deletion operation."
            ),
            "blocked": False,
        },
        {
            "pattern": r"\btaskkill\b",
            "risk": "medium",
            "reason": (
                "Process termination operation."
            ),
            "blocked": False,
        },
        {
            "pattern": r"\bkill\b",
            "risk": "medium",
            "reason": (
                "Process termination operation."
            ),
            "blocked": False,
        },
        {
            "pattern": r"\breg\s+delete\b",
            "risk": "high",
            "reason": (
                "Registry deletion operation."
            ),
            "blocked": False,
        },
        {
            "pattern": r"\bpower\s*shell\b",
            "risk": "low",
            "reason": (
                "PowerShell command execution."
            ),
            "blocked": False,
        },
        {
            "pattern": r"\bcurl\b.*\|\s*(sh|bash)",
            "risk": "critical",
            "reason": (
                "Remote script piped directly "
                "to shell."
            ),
            "blocked": True,
        },
        {
            "pattern": r"\bwget\b.*\|\s*(sh|bash)",
            "risk": "critical",
            "reason": (
                "Remote script piped directly "
                "to shell."
            ),
            "blocked": True,
        },
    ]

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize command safety system.
        """

        self.config = config or {}

        self.enabled = self.config.get(
            "enabled",
            True,
        )

        self.block_critical = bool(
            self.config.get(
                "block_critical",
                True,
            )
        )

        self.confirm_medium = bool(
            self.config.get(
                "confirm_medium",
                False,
            )
        )

        self.confirm_high = bool(
            self.config.get(
                "confirm_high",
                True,
            )
        )

        self.confirm_critical = bool(
            self.config.get(
                "confirm_critical",
                True,
            )
        )

        self.patterns = list(
            self.DEFAULT_DANGEROUS_PATTERNS
        )

        self.custom_rules: List[
            Dict[str, Any]
        ] = []

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
            "CommandSafety initialized."
        )

    # ============================================================
    # ADD SAFETY PATTERN
    # ============================================================

    def add_pattern(
        self,
        pattern: str,
        risk: str = "medium",
        reason: str = (
            "Custom safety rule matched."
        ),
        blocked: bool = False,
    ) -> bool:
        """
        Add a command safety pattern.
        """

        if risk not in self.RISK_LEVELS:
            return False

        try:
            re.compile(
                pattern,
                re.IGNORECASE,
            )

        except re.error:

            logger.error(
                "Invalid regex pattern: %s",
                pattern,
            )

            return False

        self.custom_rules.append(
            {
                "pattern": pattern,
                "risk": risk,
                "reason": reason,
                "blocked": blocked,
            }
        )

        logger.info(
            "Custom command safety rule added."
        )

        return True

    # ============================================================
    # REMOVE CUSTOM RULE
    # ============================================================

    def remove_pattern(
        self,
        pattern: str,
    ) -> bool:
        """
        Remove a custom safety rule.
        """

        for rule in self.custom_rules:

            if rule["pattern"] == pattern:

                self.custom_rules.remove(
                    rule
                )

                return True

        return False

    # ============================================================
    # CALCULATE RISK
    # ============================================================

    def _get_highest_risk(
        self,
        risks: List[str],
    ) -> str:
        """
        Return highest detected risk level.
        """

        if not risks:
            return "safe"

        return max(
            risks,
            key=lambda risk: (
                self.RISK_LEVELS.get(
                    risk,
                    0,
                )
            ),
        )

    # ============================================================
    # PATTERN ANALYSIS
    # ============================================================

    def _analyze_patterns(
        self,
        command: str,
    ) -> List[Dict[str, Any]]:
        """
        Match command against safety rules.
        """

        matches = []

        all_patterns = (
            self.patterns
            + self.custom_rules
        )

        for rule in all_patterns:

            pattern = rule.get(
                "pattern",
                ""
            )

            try:

                matched = re.search(
                    pattern,
                    command,
                    re.IGNORECASE,
                )

            except re.error:

                logger.error(
                    "Invalid safety pattern: %s",
                    pattern,
                )

                continue

            if matched:

                matches.append(
                    {
                        "pattern": pattern,
                        "risk": rule.get(
                            "risk",
                            "medium",
                        ),
                        "reason": rule.get(
                            "reason",
                            "Safety rule matched.",
                        ),
                        "blocked": bool(
                            rule.get(
                                "blocked",
                                False,
                            )
                        ),
                    }
                )

        return matches

    # ============================================================
    # CONFIRMATION REQUIREMENT
    # ============================================================

    def _requires_confirmation(
        self,
        risk_level: str,
    ) -> bool:
        """
        Determine whether confirmation
        is required.
        """

        if risk_level == "medium":

            return self.confirm_medium

        if risk_level == "high":

            return self.confirm_high

        if risk_level == "critical":

            return self.confirm_critical

        return False

    # ============================================================
    # ANALYZE COMMAND
    # ============================================================

    def analyze(
        self,
        command: str,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a command before execution.

        Returns:

            allowed
            risk_level
            requires_confirmation
            blocked
            reasons
            matches
        """

        command = (
            command
            or ""
        ).strip()

        result = {
            "command": command,
            "allowed": True,
            "blocked": False,
            "risk_level": "safe",
            "risk_score": 0,
            "requires_confirmation": False,
            "reasons": [],
            "matches": [],
            "context": context or {},
            "timestamp": (
                datetime.now().isoformat()
            ),
        }

        # Empty commands are invalid.
        if not command:

            result["allowed"] = False
            result["risk_level"] = "medium"
            result["risk_score"] = (
                self.RISK_LEVELS["medium"]
            )
            result["reasons"].append(
                "Empty command."
            )

            self._record_history(
                result
            )

            return result

        # Safety disabled.
        if not self.enabled:

            result["reasons"].append(
                "Command safety disabled."
            )

            self._record_history(
                result
            )

            return result

        matches = self._analyze_patterns(
            command
        )

        risks = [
            match["risk"]
            for match in matches
        ]

        highest_risk = (
            self._get_highest_risk(
                risks
            )
        )

        blocked = any(
            match["blocked"]
            for match in matches
        )

        result["matches"] = matches
        result["risk_level"] = (
            highest_risk
        )
        result["risk_score"] = (
            self.RISK_LEVELS[
                highest_risk
            ]
        )

        result["reasons"] = [
            match["reason"]
            for match in matches
        ]

        # Explicitly blocked commands.
        if blocked:

            result["blocked"] = True
            result["allowed"] = False

            result["requires_confirmation"] = (
                False
            )

            result["reasons"].append(
                "Command blocked by security policy."
            )

        # Critical commands.
        elif (
            highest_risk == "critical"
        ):

            if self.block_critical:

                result["blocked"] = True
                result["allowed"] = False

                result["reasons"].append(
                    "Critical command blocked."
                )

            else:

                result[
                    "requires_confirmation"
                ] = (
                    self._requires_confirmation(
                        highest_risk
                    )
                )

        # High and medium commands.
        else:

            result[
                "requires_confirmation"
            ] = (
                self._requires_confirmation(
                    highest_risk
                )
            )

        self._record_history(
            result
        )

        logger.info(
            "Command analyzed | "
            "Risk: %s | "
            "Allowed: %s",
            highest_risk,
            result["allowed"],
        )

        return result

    # ============================================================
    # SIMPLE CHECK
    # ============================================================

    def is_safe(
        self,
        command: str,
    ) -> bool:
        """
        Return True if command is safe
        and does not require confirmation.
        """

        result = self.analyze(
            command
        )

        return (
            result["allowed"]
            and not result[
                "requires_confirmation"
            ]
            and result["risk_level"]
            in (
                "safe",
                "low",
            )
        )

    # ============================================================
    # EXECUTION CHECK
    # ============================================================

    def can_execute(
        self,
        command: str,
        confirmed: bool = False,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Determine whether command can execute.

        Args:
            command:
                Command to analyze.

            confirmed:
                Whether the user has already
                confirmed the action.
        """

        result = self.analyze(
            command,
            context=context,
        )

        # Already blocked.
        if result["blocked"]:

            result["can_execute"] = False

            return result

        # Confirmation required.
        if (
            result[
                "requires_confirmation"
            ]
            and not confirmed
        ):

            result["can_execute"] = False

            result["reasons"].append(
                "User confirmation required."
            )

            return result

        result["can_execute"] = (
            result["allowed"]
        )

        return result

    # ============================================================
    # HISTORY
    # ============================================================

    def _record_history(
        self,
        result: Dict[str, Any],
    ) -> None:
        """
        Store command analysis history.
        """

        self.history.append(
            result.copy()
        )

        if (
            len(self.history)
            > self.max_history
        ):

            overflow = (
                len(self.history)
                - self.max_history
            )

            self.history = self.history[
                overflow:
            ]

    def get_history(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Return command safety history.
        """

        return self.history[
            -max(1, limit):
        ]

    def clear_history(
        self,
    ) -> None:
        """
        Clear command history.
        """

        self.history.clear()

    # ============================================================
    # RISK INFORMATION
    # ============================================================

    def get_risk_score(
        self,
        risk_level: str,
    ) -> int:
        """
        Return numerical risk score.
        """

        return (
            self.RISK_LEVELS.get(
                risk_level,
                0,
            )
        )

    def get_risk_levels(
        self,
    ) -> Dict[str, int]:
        """
        Return available risk levels.
        """

        return (
            self.RISK_LEVELS.copy()
        )

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return command safety status.
        """

        return {
            "enabled": self.enabled,
            "default_patterns": len(
                self.patterns
            ),
            "custom_patterns": len(
                self.custom_rules
            ),
            "history_entries": len(
                self.history
            ),
            "block_critical": (
                self.block_critical
            ),
            "confirm_medium": (
                self.confirm_medium
            ),
            "confirm_high": (
                self.confirm_high
            ),
            "confirm_critical": (
                self.confirm_critical
            ),
        }

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Safely shut down command safety system.
        """

        logger.info(
            "Shutting down CommandSafety..."
        )

        self.clear_history()
        self.custom_rules.clear()

        logger.info(
            "CommandSafety shutdown complete."
        )
