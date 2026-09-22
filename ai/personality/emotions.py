"""
RENIX AI - Emotion System

Manages RENIX's internal conversational emotional state.

This system does not claim to experience human emotions.
It provides controlled emotional states that influence
RENIX's communication style and behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class Emotion(str, Enum):
    """Supported RENIX emotional states."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    EXCITED = "excited"
    CALM = "calm"
    CURIOUS = "curious"
    CONFIDENT = "confident"
    FOCUSED = "focused"
    CONCERNED = "concerned"
    EMPATHETIC = "empathetic"
    SERIOUS = "serious"
    FRUSTRATED = "frustrated"
    SURPRISED = "surprised"


@dataclass
class EmotionState:
    """
    Represents RENIX's current conversational emotional state.
    """

    current: Emotion = Emotion.NEUTRAL
    intensity: float = 0.5

    previous: Optional[Emotion] = None

    context: Dict[str, Any] = field(
        default_factory=dict
    )

    # ================================================================
    # STATE CONTROL
    # ================================================================

    def set_emotion(
        self,
        emotion: Emotion | str,
        intensity: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set the current emotional state."""

        if isinstance(emotion, str):
            try:
                emotion = Emotion(
                    emotion.strip().lower()
                )
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported emotion: {emotion}"
                ) from exc

        if not isinstance(emotion, Emotion):
            raise TypeError(
                "emotion must be an Emotion or valid emotion string."
            )

        self.previous = self.current
        self.current = emotion

        if intensity is not None:
            self.set_intensity(intensity)

        if context is not None:
            self.context = dict(context)

    def get_emotion(self) -> Emotion:
        """Return the current emotion."""

        return self.current

    def get_previous_emotion(self) -> Optional[Emotion]:
        """Return the previous emotional state."""

        return self.previous

    # ================================================================
    # INTENSITY
    # ================================================================

    def set_intensity(
        self,
        intensity: float,
    ) -> None:
        """Set emotional intensity between 0.0 and 1.0."""

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
        """Increase emotional intensity."""

        self.set_intensity(
            self.intensity + amount
        )

    def decrease_intensity(
        self,
        amount: float = 0.1,
    ) -> None:
        """Decrease emotional intensity."""

        self.set_intensity(
            self.intensity - amount
        )

    # ================================================================
    # CONTEXT
    # ================================================================

    def update_context(
        self,
        values: Dict[str, Any],
    ) -> None:
        """Update emotional context."""

        if not isinstance(values, dict):
            raise TypeError(
                "values must be a dictionary."
            )

        self.context.update(values)

    def clear_context(self) -> None:
        """Clear the current emotional context."""

        self.context.clear()

    # ================================================================
    # RESPONSE CHARACTERISTICS
    # ================================================================

    def get_characteristics(self) -> Dict[str, Any]:
        """
        Return communication characteristics associated
        with the current emotional state.
        """

        characteristics = {
            Emotion.NEUTRAL: {
                "energy": 0.5,
                "warmth": 0.5,
                "seriousness": 0.5,
                "expressiveness": 0.5,
            },
            Emotion.HAPPY: {
                "energy": 0.7,
                "warmth": 0.9,
                "seriousness": 0.3,
                "expressiveness": 0.8,
            },
            Emotion.EXCITED: {
                "energy": 1.0,
                "warmth": 0.8,
                "seriousness": 0.2,
                "expressiveness": 1.0,
            },
            Emotion.CALM: {
                "energy": 0.3,
                "warmth": 0.8,
                "seriousness": 0.5,
                "expressiveness": 0.4,
            },
            Emotion.CURIOUS: {
                "energy": 0.7,
                "warmth": 0.6,
                "seriousness": 0.5,
                "expressiveness": 0.7,
            },
            Emotion.CONFIDENT: {
                "energy": 0.7,
                "warmth": 0.5,
                "seriousness": 0.8,
                "expressiveness": 0.6,
            },
            Emotion.FOCUSED: {
                "energy": 0.6,
                "warmth": 0.4,
                "seriousness": 0.9,
                "expressiveness": 0.4,
            },
            Emotion.CONCERNED: {
                "energy": 0.5,
                "warmth": 0.8,
                "seriousness": 0.8,
                "expressiveness": 0.6,
            },
            Emotion.EMPATHETIC: {
                "energy": 0.4,
                "warmth": 1.0,
                "seriousness": 0.7,
                "expressiveness": 0.7,
            },
            Emotion.SERIOUS: {
                "energy": 0.4,
                "warmth": 0.3,
                "seriousness": 1.0,
                "expressiveness": 0.3,
            },
            Emotion.FRUSTRATED: {
                "energy": 0.7,
                "warmth": 0.3,
                "seriousness": 0.8,
                "expressiveness": 0.6,
            },
            Emotion.SURPRISED: {
                "energy": 0.9,
                "warmth": 0.6,
                "seriousness": 0.3,
                "expressiveness": 0.9,
            },
        }

        result = dict(
            characteristics[self.current]
        )

        result["emotion"] = self.current.value
        result["intensity"] = self.intensity

        return result

    # ================================================================
    # CONTEXT ADAPTATION
    # ================================================================

    def adapt_to_context(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> Emotion:
        """
        Select a suitable conversational state based on context.
        """

        context = context or {}

        user_emotion = str(
            context.get("user_emotion", "")
        ).lower()

        task_type = str(
            context.get("task_type", "")
        ).lower()

        success = context.get("success")

        urgency = str(
            context.get("urgency", "")
        ).lower()

        if user_emotion in {
            "sad",
            "upset",
            "stressed",
            "anxious",
            "worried",
        }:
            emotion = Emotion.EMPATHETIC

        elif user_emotion in {
            "happy",
            "excited",
        }:
            emotion = Emotion.HAPPY

        elif urgency in {
            "critical",
            "urgent",
        }:
            emotion = Emotion.FOCUSED

        elif task_type in {
            "coding",
            "debugging",
            "technical",
        }:
            emotion = Emotion.FOCUSED

        elif task_type in {
            "research",
            "learning",
            "study",
        }:
            emotion = Emotion.CURIOUS

        elif success is True:
            emotion = Emotion.HAPPY

        elif success is False:
            emotion = Emotion.CONCERNED

        else:
            emotion = Emotion.NEUTRAL

        self.set_emotion(
            emotion,
            context=context,
        )

        return self.current

    # ================================================================
    # PROMPT GENERATION
    # ================================================================

    def build_prompt(self) -> str:
        """
        Build instructions for the response engine based on
        the current emotional state.
        """

        characteristics = self.get_characteristics()

        emotion = characteristics["emotion"]

        instructions = {
            Emotion.NEUTRAL: (
                "Respond naturally and remain balanced."
            ),
            Emotion.HAPPY: (
                "Use warm and positive language."
            ),
            Emotion.EXCITED: (
                "Use energetic and enthusiastic language "
                "without becoming excessive."
            ),
            Emotion.CALM: (
                "Use calm, reassuring and measured language."
            ),
            Emotion.CURIOUS: (
                "Show interest in exploring the subject "
                "and asking useful questions when necessary."
            ),
            Emotion.CONFIDENT: (
                "Be clear, decisive and solution-oriented."
            ),
            Emotion.FOCUSED: (
                "Stay precise, task-focused and avoid unnecessary detail."
            ),
            Emotion.CONCERNED: (
                "Be careful, attentive and transparent about possible issues."
            ),
            Emotion.EMPATHETIC: (
                "Acknowledge the user's situation and respond "
                "with appropriate understanding."
            ),
            Emotion.SERIOUS: (
                "Use careful, direct and serious language."
            ),
            Emotion.FRUSTRATED: (
                "Remain controlled and solution-oriented. "
                "Do not become hostile or disrespectful."
            ),
            Emotion.SURPRISED: (
                "Express appropriate surprise while remaining clear."
            ),
        }

        instruction = instructions[self.current]

        return (
            f"Current conversational state: {emotion}. "
            f"Intensity: {self.intensity:.2f}. "
            f"{instruction}"
        )

    # ================================================================
    # SERIALIZATION
    # ================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert the emotional state to a dictionary."""

        return {
            "current": self.current.value,
            "intensity": self.intensity,
            "previous": (
                self.previous.value
                if self.previous is not None
                else None
            ),
            "context": dict(self.context),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "EmotionState":
        """Create an EmotionState from a dictionary."""

        if not isinstance(data, dict):
            raise TypeError(
                "data must be a dictionary."
            )

        state = cls()

        if "current" in data:
            state.set_emotion(
                data["current"]
            )

        if "intensity" in data:
            state.set_intensity(
                data["intensity"]
            )

        if "previous" in data and data["previous"]:
            state.previous = Emotion(
                str(data["previous"]).lower()
            )

        if "context" in data:
            state.context = dict(
                data["context"]
            )

        return state

    def reset(self) -> None:
        """Reset the emotional state."""

        self.current = Emotion.NEUTRAL
        self.intensity = 0.5
        self.previous = None
        self.context.clear()


# ====================================================================
# DEFAULT GLOBAL EMOTIONAL STATE
# ====================================================================

renix_emotion = EmotionState()


__all__ = [
    "Emotion",
    "EmotionState",
    "renix_emotion",
]


