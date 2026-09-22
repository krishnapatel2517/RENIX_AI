"""
RENIX AI - Tone System

Controls how RENIX communicates with the user.

The personality system defines WHO RENIX is.
The tone system defines HOW RENIX communicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ToneStyle(str, Enum):
    """Available communication tones."""

    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    CONFIDENT = "confident"
    EMPATHETIC = "empathetic"
    HUMOROUS = "humorous"
    ENCOURAGING = "encouraging"
    SERIOUS = "serious"
    CONCISE = "concise"
    EXCITED = "excited"


@dataclass
class Tone:
    """
    RENIX communication-tone controller.

    Tone can be changed dynamically depending on context,
    emotion, task type, or user preference.
    """

    current: ToneStyle = ToneStyle.FRIENDLY

    intensity: float = 0.7

    formal: bool = False
    expressive: bool = True
    concise: bool = False

    custom_settings: Dict[str, Any] = field(
        default_factory=dict
    )

    # ================================================================
    # TONE CONTROL
    # ================================================================

    def set_tone(
        self,
        tone: ToneStyle | str,
    ) -> None:
        """Set the current communication tone."""

        if isinstance(tone, str):
            try:
                tone = ToneStyle(tone.lower().strip())
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported tone: {tone}"
                ) from exc

        if not isinstance(tone, ToneStyle):
            raise TypeError(
                "tone must be a ToneStyle or valid tone string."
            )

        self.current = tone

    def get_tone(self) -> ToneStyle:
        """Return the current tone."""

        return self.current

    # ================================================================
    # INTENSITY
    # ================================================================

    def set_intensity(
        self,
        intensity: float,
    ) -> None:
        """
        Set tone intensity between 0.0 and 1.0.
        """

        try:
            intensity = float(intensity)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "intensity must be a number."
            ) from exc

        self.intensity = max(
            0.0,
            min(1.0, intensity),
        )

    def increase_intensity(
        self,
        amount: float = 0.1,
    ) -> None:
        """Increase tone intensity."""

        self.set_intensity(
            self.intensity + amount
        )

    def decrease_intensity(
        self,
        amount: float = 0.1,
    ) -> None:
        """Decrease tone intensity."""

        self.set_intensity(
            self.intensity - amount
        )

    # ================================================================
    # COMMUNICATION SETTINGS
    # ================================================================

    def enable_formal_mode(self) -> None:
        """Enable formal communication."""

        self.formal = True

    def disable_formal_mode(self) -> None:
        """Disable formal communication."""

        self.formal = False

    def enable_expressive_mode(self) -> None:
        """Enable expressive communication."""

        self.expressive = True

    def disable_expressive_mode(self) -> None:
        """Disable expressive communication."""

        self.expressive = False

    def enable_concise_mode(self) -> None:
        """Enable concise responses."""

        self.concise = True

    def disable_concise_mode(self) -> None:
        """Disable concise responses."""

        self.concise = False

    # ================================================================
    # CONTEXT ADAPTATION
    # ================================================================

    def adapt_to_context(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToneStyle:
        """
        Automatically select a suitable tone based on context.
        """

        context = context or {}

        task_type = str(
            context.get("task_type", "")
        ).lower()

        emotional_state = str(
            context.get("emotion", "")
        ).lower()

        urgency = str(
            context.get("urgency", "")
        ).lower()

        if emotional_state in {
            "sad",
            "upset",
            "stressed",
            "anxious",
            "angry",
        }:
            self.current = ToneStyle.EMPATHETIC

        elif urgency in {
            "critical",
            "urgent",
            "high",
        }:
            self.current = ToneStyle.CONFIDENT

        elif task_type in {
            "coding",
            "debugging",
            "technical",
            "system",
        }:
            self.current = ToneStyle.PROFESSIONAL

        elif task_type in {
            "study",
            "education",
            "learning",
        }:
            self.current = ToneStyle.ENCOURAGING

        elif task_type in {
            "casual",
            "conversation",
            "chat",
        }:
            self.current = ToneStyle.FRIENDLY

        return self.current

    # ================================================================
    # RESPONSE GUIDANCE
    # ================================================================

    def get_response_guidelines(self) -> Dict[str, Any]:
        """Return instructions describing the current tone."""

        guidelines: Dict[str, Any] = {
            "tone": self.current.value,
            "intensity": self.intensity,
            "formal": self.formal,
            "expressive": self.expressive,
            "concise": self.concise,
        }

        tone_guidelines = {
            ToneStyle.NEUTRAL: (
                "Use balanced, clear and objective language."
            ),

            ToneStyle.FRIENDLY: (
                "Be warm, approachable and natural."
            ),

            ToneStyle.PROFESSIONAL: (
                "Be precise, structured and professional."
            ),

            ToneStyle.CASUAL: (
                "Use relaxed and conversational language."
            ),

            ToneStyle.CONFIDENT: (
                "Be decisive, clear and solution-oriented."
            ),

            ToneStyle.EMPATHETIC: (
                "Acknowledge the user's feelings and respond "
                "with appropriate understanding."
            ),

            ToneStyle.HUMOROUS: (
                "Use light and appropriate humor when suitable."
            ),

            ToneStyle.ENCOURAGING: (
                "Be supportive and motivating while remaining "
                "realistic."
            ),

            ToneStyle.SERIOUS: (
                "Use focused, direct and careful language."
            ),

            ToneStyle.CONCISE: (
                "Give the necessary information with minimal "
                "unnecessary wording."
            ),

            ToneStyle.EXCITED: (
                "Use energetic and enthusiastic language "
                "without becoming excessive."
            ),
        }

        guidelines["instruction"] = tone_guidelines[
            self.current
        ]

        return guidelines

    def build_prompt(self) -> str:
        """
        Build a concise instruction for an AI response engine.
        """

        guidelines = self.get_response_guidelines()

        parts = [
            f"Use a {guidelines['tone']} tone.",
            guidelines["instruction"],
        ]

        if guidelines["formal"]:
            parts.append(
                "Maintain formal language."
            )

        if guidelines["expressive"]:
            parts.append(
                "Allow natural emotional expression."
            )
        else:
            parts.append(
                "Keep emotional expression restrained."
            )

        if guidelines["concise"]:
            parts.append(
                "Keep the response concise."
            )

        return " ".join(parts)

    # ================================================================
    # CUSTOM SETTINGS
    # ================================================================

    def set_setting(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Set a custom tone setting."""

        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                "Setting key cannot be empty."
            )

        self.custom_settings[key.strip()] = value

    def get_setting(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get a custom tone setting."""

        return self.custom_settings.get(
            key,
            default,
        )

    # ================================================================
    # SERIALIZATION
    # ================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert tone configuration to a dictionary."""

        return {
            "current": self.current.value,
            "intensity": self.intensity,
            "formal": self.formal,
            "expressive": self.expressive,
            "concise": self.concise,
            "custom_settings": dict(
                self.custom_settings
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "Tone":
        """Create a Tone object from a dictionary."""

        if not isinstance(data, dict):
            raise TypeError(
                "data must be a dictionary."
            )

        tone = cls()

        if "current" in data:
            tone.set_tone(data["current"])

        if "intensity" in data:
            tone.set_intensity(
                data["intensity"]
            )

        if "formal" in data:
            tone.formal = bool(
                data["formal"]
            )

        if "expressive" in data:
            tone.expressive = bool(
                data["expressive"]
            )

        if "concise" in data:
            tone.concise = bool(
                data["concise"]
            )

        if "custom_settings" in data:
            tone.custom_settings = dict(
                data["custom_settings"]
            )

        return tone

    def reset(self) -> None:
        """Reset tone configuration to defaults."""

        default = Tone()

        self.current = default.current
        self.intensity = default.intensity
        self.formal = default.formal
        self.expressive = default.expressive
        self.concise = default.concise
        self.custom_settings.clear()


# ====================================================================
# DEFAULT TONE INSTANCE
# ====================================================================

renix_tone = Tone()


__all__ = [
    "ToneStyle",
    "Tone",
    "renix_tone",
]


