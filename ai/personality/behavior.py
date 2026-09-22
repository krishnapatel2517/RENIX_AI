"""
RENIX AI - Behavior System

Controls how RENIX behaves during conversations and tasks.

Personality = who RENIX is
Tone        = how RENIX speaks
Emotion     = current conversational state
Behavior    = how RENIX acts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class BehaviorMode(str, Enum):
    """Available RENIX behavior modes."""

    NORMAL = "normal"
    ASSISTIVE = "assistive"
    FOCUSED = "focused"
    PROACTIVE = "proactive"
    CAUTIOUS = "cautious"
    CREATIVE = "creative"
    TEACHING = "teaching"
    PROBLEM_SOLVING = "problem_solving"
    AUTOMATION = "automation"
    EMERGENCY = "emergency"


@dataclass
class Behavior:
    """
    Controls RENIX's behavioral decision preferences.

    This class does not execute system actions itself.
    It provides behavioral instructions to the orchestration
    and execution layers.
    """

    mode: BehaviorMode = BehaviorMode.NORMAL

    proactive: bool = True
    ask_confirmation: bool = True
    explain_actions: bool = True
    learn_from_interactions: bool = True
    adapt_to_context: bool = True

    risk_tolerance: float = 0.3
    initiative: float = 0.7

    preferences: Dict[str, Any] = field(
        default_factory=dict
    )

    # ================================================================
    # MODE CONTROL
    # ================================================================

    def set_mode(
        self,
        mode: BehaviorMode | str,
    ) -> None:
        """Set RENIX's current behavior mode."""

        if isinstance(mode, str):
            try:
                mode = BehaviorMode(
                    mode.strip().lower()
                )
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported behavior mode: {mode}"
                ) from exc

        if not isinstance(mode, BehaviorMode):
            raise TypeError(
                "mode must be a BehaviorMode or valid mode string."
            )

        self.mode = mode

    def get_mode(self) -> BehaviorMode:
        """Return the current behavior mode."""

        return self.mode

    # ================================================================
    # PROACTIVITY
    # ================================================================

    def enable_proactive_behavior(self) -> None:
        """Allow RENIX to proactively assist when appropriate."""

        self.proactive = True

    def disable_proactive_behavior(self) -> None:
        """Disable proactive behavior."""

        self.proactive = False

    # ================================================================
    # CONFIRMATION
    # ================================================================

    def enable_confirmation(self) -> None:
        """Require confirmation for actions configured as confirmable."""

        self.ask_confirmation = True

    def disable_confirmation(self) -> None:
        """
        Disable the behavioral confirmation preference.

        Actual security-sensitive actions should still be governed
        by the security package.
        """

        self.ask_confirmation = False

    # ================================================================
    # EXPLANATIONS
    # ================================================================

    def enable_action_explanations(self) -> None:
        """Enable explanations before/after important actions."""

        self.explain_actions = True

    def disable_action_explanations(self) -> None:
        """Disable additional action explanations."""

        self.explain_actions = False

    # ================================================================
    # LEARNING
    # ================================================================

    def enable_learning(self) -> None:
        """Allow the higher-level memory system to learn preferences."""

        self.learn_from_interactions = True

    def disable_learning(self) -> None:
        """Disable behavioral learning preference."""

        self.learn_from_interactions = False

    # ================================================================
    # ADAPTATION
    # ================================================================

    def enable_adaptation(self) -> None:
        """Allow behavior to adapt to context."""

        self.adapt_to_context = True

    def disable_adaptation(self) -> None:
        """Disable contextual behavior adaptation."""

        self.adapt_to_context = False

    # ================================================================
    # RISK
    # ================================================================

    def set_risk_tolerance(
        self,
        value: float,
    ) -> None:
        """Set risk tolerance between 0.0 and 1.0."""

        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "risk tolerance must be a number."
            ) from exc

        self.risk_tolerance = max(
            0.0,
            min(1.0, value),
        )

    def set_initiative(
        self,
        value: float,
    ) -> None:
        """Set initiative level between 0.0 and 1.0."""

        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "initiative must be a number."
            ) from exc

        self.initiative = max(
            0.0,
            min(1.0, value),
        )

    # ================================================================
    # CONTEXT ADAPTATION
    # ================================================================

    def adapt(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> BehaviorMode:
        """
        Select an appropriate behavior mode from context.
        """

        if not self.adapt_to_context:
            return self.mode

        context = context or {}

        task_type = str(
            context.get("task_type", "")
        ).lower()

        urgency = str(
            context.get("urgency", "")
        ).lower()

        user_requested = str(
            context.get("request_type", "")
        ).lower()

        if urgency in {
            "critical",
            "emergency",
        }:
            self.mode = BehaviorMode.EMERGENCY

        elif task_type in {
            "automation",
            "workflow",
            "routine",
        }:
            self.mode = BehaviorMode.AUTOMATION

        elif task_type in {
            "teaching",
            "study",
            "education",
            "learning",
        }:
            self.mode = BehaviorMode.TEACHING

        elif task_type in {
            "creative",
            "writing",
            "design",
        }:
            self.mode = BehaviorMode.CREATIVE

        elif task_type in {
            "problem_solving",
            "debugging",
            "analysis",
        }:
            self.mode = BehaviorMode.PROBLEM_SOLVING

        elif task_type in {
            "system",
            "security",
            "sensitive",
        }:
            self.mode = BehaviorMode.CAUTIOUS

        elif user_requested in {
            "suggestion",
            "recommendation",
            "proactive",
        }:
            self.mode = BehaviorMode.PROACTIVE

        else:
            self.mode = BehaviorMode.NORMAL

        return self.mode

    # ================================================================
    # ACTION DECISION
    # ================================================================

    def should_confirm(
        self,
        action: str,
        risk_level: float = 0.0,
    ) -> bool:
        """
        Determine whether the behavior layer recommends confirmation.

        The security layer remains authoritative for actual
        permission and safety decisions.
        """

        if not self.ask_confirmation:
            return False

        try:
            risk_level = float(risk_level)
        except (TypeError, ValueError):
            risk_level = 0.0

        risk_level = max(
            0.0,
            min(1.0, risk_level),
        )

        sensitive_keywords = {
            "delete",
            "remove",
            "format",
            "shutdown",
            "restart",
            "payment",
            "purchase",
            "send",
            "publish",
            "install",
            "uninstall",
            "execute",
        }

        action_lower = str(action).lower()

        if risk_level >= 0.5:
            return True

        return any(
            keyword in action_lower
            for keyword in sensitive_keywords
        )

    def should_be_proactive(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Determine whether RENIX should proactively offer assistance.
        """

        if not self.proactive:
            return False

        context = context or {}

        user_busy = bool(
            context.get("user_busy", False)
        )

        explicit_request = bool(
            context.get("explicit_request", False)
        )

        if user_busy and not explicit_request:
            return False

        return self.initiative >= 0.5

    # ================================================================
    # BEHAVIOR GUIDELINES
    # ================================================================

    def get_guidelines(self) -> Dict[str, Any]:
        """Return behavior instructions for other RENIX components."""

        mode_instructions = {
            BehaviorMode.NORMAL: (
                "Act naturally and provide direct assistance."
            ),

            BehaviorMode.ASSISTIVE: (
                "Prioritize helping the user complete the requested task."
            ),

            BehaviorMode.FOCUSED: (
                "Stay focused on the current objective and minimize distractions."
            ),

            BehaviorMode.PROACTIVE: (
                "Anticipate useful next steps without becoming intrusive."
            ),

            BehaviorMode.CAUTIOUS: (
                "Verify important assumptions and avoid risky actions."
            ),

            BehaviorMode.CREATIVE: (
                "Explore useful alternatives and generate creative solutions."
            ),

            BehaviorMode.TEACHING: (
                "Explain concepts clearly and guide the user step by step."
            ),

            BehaviorMode.PROBLEM_SOLVING: (
                "Analyze the problem systematically and prioritize practical solutions."
            ),

            BehaviorMode.AUTOMATION: (
                "Execute automation tasks in a controlled and traceable manner."
            ),

            BehaviorMode.EMERGENCY: (
                "Prioritize safety, clarity and controlled actions."
            ),
        }

        return {
            "mode": self.mode.value,
            "instruction": mode_instructions[self.mode],
            "proactive": self.proactive,
            "ask_confirmation": self.ask_confirmation,
            "explain_actions": self.explain_actions,
            "learn_from_interactions": self.learn_from_interactions,
            "adapt_to_context": self.adapt_to_context,
            "risk_tolerance": self.risk_tolerance,
            "initiative": self.initiative,
        }

    def build_prompt(self) -> str:
        """Build behavioral instructions for the AI response layer."""

        guidelines = self.get_guidelines()

        parts: List[str] = [
            f"Behavior mode: {guidelines['mode']}.",
            guidelines["instruction"],
        ]

        if guidelines["proactive"]:
            parts.append(
                "Offer useful next steps when appropriate."
            )

        if guidelines["explain_actions"]:
            parts.append(
                "Explain important actions clearly."
            )

        if guidelines["ask_confirmation"]:
            parts.append(
                "Request confirmation when an action is potentially risky."
            )

        return " ".join(parts)

    # ================================================================
    # CUSTOM PREFERENCES
    # ================================================================

    def set_preference(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Set a custom behavior preference."""

        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                "Preference key cannot be empty."
            )

        self.preferences[key.strip()] = value

    def get_preference(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get a custom behavior preference."""

        return self.preferences.get(
            key,
            default,
        )

    def remove_preference(
        self,
        key: str,
    ) -> bool:
        """Remove a custom behavior preference."""

        if key in self.preferences:
            del self.preferences[key]
            return True

        return False

    # ================================================================
    # SERIALIZATION
    # ================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert behavior configuration to a dictionary."""

        return {
            "mode": self.mode.value,
            "proactive": self.proactive,
            "ask_confirmation": self.ask_confirmation,
            "explain_actions": self.explain_actions,
            "learn_from_interactions": self.learn_from_interactions,
            "adapt_to_context": self.adapt_to_context,
            "risk_tolerance": self.risk_tolerance,
            "initiative": self.initiative,
            "preferences": dict(self.preferences),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "Behavior":
        """Create a Behavior object from a dictionary."""

        if not isinstance(data, dict):
            raise TypeError(
                "data must be a dictionary."
            )

        behavior = cls()

        if "mode" in data:
            behavior.set_mode(data["mode"])

        if "proactive" in data:
            behavior.proactive = bool(
                data["proactive"]
            )

        if "ask_confirmation" in data:
            behavior.ask_confirmation = bool(
                data["ask_confirmation"]
            )

        if "explain_actions" in data:
            behavior.explain_actions = bool(
                data["explain_actions"]
            )

        if "learn_from_interactions" in data:
            behavior.learn_from_interactions = bool(
                data["learn_from_interactions"]
            )

        if "adapt_to_context" in data:
            behavior.adapt_to_context = bool(
                data["adapt_to_context"]
            )

        if "risk_tolerance" in data:
            behavior.set_risk_tolerance(
                data["risk_tolerance"]
            )

        if "initiative" in data:
            behavior.set_initiative(
                data["initiative"]
            )

        if "preferences" in data:
            behavior.preferences = dict(
                data["preferences"]
            )

        return behavior

    def reset(self) -> None:
        """Reset behavior settings to defaults."""

        default = Behavior()

        self.mode = default.mode
        self.proactive = default.proactive
        self.ask_confirmation = default.ask_confirmation
        self.explain_actions = default.explain_actions
        self.learn_from_interactions = default.learn_from_interactions
        self.adapt_to_context = default.adapt_to_context
        self.risk_tolerance = default.risk_tolerance
        self.initiative = default.initiative
        self.preferences.clear()


# ====================================================================
# DEFAULT GLOBAL BEHAVIOR INSTANCE
# ====================================================================

renix_behavior = Behavior()


__all__ = [
    "BehaviorMode",
    "Behavior",
    "renix_behavior",
]


