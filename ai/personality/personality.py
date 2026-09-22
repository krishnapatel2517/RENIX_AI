"""
RENIX AI - Personality System

Defines RENIX's core personality configuration and identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Personality:
    """
    Core personality configuration for RENIX.

    This class stores stable personality characteristics while
    allowing the behavior and tone systems to adapt responses.
    """

    name: str = "RENIX"

    identity: str = (
        "RENIX is an intelligent personal AI assistant designed "
        "to communicate naturally, assist with tasks, and adapt "
        "to the user's context."
    )

    traits: List[str] = field(
        default_factory=lambda: [
            "intelligent",
            "helpful",
            "calm",
            "respectful",
            "confident",
            "curious",
            "adaptive",
            "honest",
            "practical",
        ]
    )

    communication_style: str = "natural"

    humor_enabled: bool = True
    empathy_enabled: bool = True
    concise_mode: bool = False

    custom_attributes: Dict[str, Any] = field(
        default_factory=dict
    )

    # ================================================================
    # TRAITS
    # ================================================================

    def add_trait(self, trait: str) -> bool:
        """Add a personality trait."""

        if not isinstance(trait, str):
            return False

        trait = trait.strip()

        if not trait:
            return False

        if trait.lower() not in {
            existing.lower()
            for existing in self.traits
        }:
            self.traits.append(trait)

        return True

    def remove_trait(self, trait: str) -> bool:
        """Remove a personality trait."""

        if not isinstance(trait, str):
            return False

        target = trait.strip().lower()

        for existing in self.traits:
            if existing.lower() == target:
                self.traits.remove(existing)
                return True

        return False

    def has_trait(self, trait: str) -> bool:
        """Check whether RENIX has a particular trait."""

        if not isinstance(trait, str):
            return False

        target = trait.strip().lower()

        return any(
            existing.lower() == target
            for existing in self.traits
        )

    # ================================================================
    # CONFIGURATION
    # ================================================================

    def set_attribute(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Store a custom personality attribute."""

        if not key or not isinstance(key, str):
            raise ValueError("Personality attribute key must be a string.")

        self.custom_attributes[key.strip()] = value

    def get_attribute(
        self,
        key: str,
        default: Optional[Any] = None,
    ) -> Any:
        """Retrieve a custom personality attribute."""

        return self.custom_attributes.get(
            key,
            default,
        )

    def set_communication_style(
        self,
        style: str,
    ) -> None:
        """Change the communication style."""

        if not isinstance(style, str):
            raise TypeError("style must be a string.")

        style = style.strip()

        if not style:
            raise ValueError("style cannot be empty.")

        self.communication_style = style

    def enable_humor(self) -> None:
        """Enable humorous responses."""

        self.humor_enabled = True

    def disable_humor(self) -> None:
        """Disable humorous responses."""

        self.humor_enabled = False

    def enable_empathy(self) -> None:
        """Enable empathetic behavior."""

        self.empathy_enabled = True

    def disable_empathy(self) -> None:
        """Disable empathetic behavior."""

        self.empathy_enabled = False

    def enable_concise_mode(self) -> None:
        """Enable concise communication."""

        self.concise_mode = True

    def disable_concise_mode(self) -> None:
        """Disable concise communication."""

        self.concise_mode = False

    # ================================================================
    # SYSTEM PROMPT / DESCRIPTION
    # ================================================================

    def build_personality_prompt(self) -> str:
        """
        Build a personality description that can be supplied to
        an AI/LLM layer.
        """

        traits = ", ".join(self.traits)

        prompt_parts = [
            f"You are {self.name}.",
            self.identity,
            f"Your core traits are: {traits}.",
            f"Your communication style is {self.communication_style}.",
        ]

        if self.humor_enabled:
            prompt_parts.append(
                "Use appropriate humor when it naturally fits the conversation."
            )
        else:
            prompt_parts.append(
                "Avoid unnecessary humor."
            )

        if self.empathy_enabled:
            prompt_parts.append(
                "Respond with appropriate empathy and emotional awareness."
            )

        if self.concise_mode:
            prompt_parts.append(
                "Prefer concise and direct responses."
            )

        return " ".join(prompt_parts)

    # ================================================================
    # SERIALIZATION
    # ================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert personality configuration to a dictionary."""

        return {
            "name": self.name,
            "identity": self.identity,
            "traits": list(self.traits),
            "communication_style": self.communication_style,
            "humor_enabled": self.humor_enabled,
            "empathy_enabled": self.empathy_enabled,
            "concise_mode": self.concise_mode,
            "custom_attributes": dict(self.custom_attributes),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "Personality":
        """Create a Personality object from a dictionary."""

        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary.")

        personality = cls()

        if "name" in data:
            personality.name = str(data["name"])

        if "identity" in data:
            personality.identity = str(data["identity"])

        if "traits" in data:
            if not isinstance(data["traits"], list):
                raise TypeError("traits must be a list.")

            personality.traits = [
                str(trait)
                for trait in data["traits"]
            ]

        if "communication_style" in data:
            personality.communication_style = str(
                data["communication_style"]
            )

        if "humor_enabled" in data:
            personality.humor_enabled = bool(
                data["humor_enabled"]
            )

        if "empathy_enabled" in data:
            personality.empathy_enabled = bool(
                data["empathy_enabled"]
            )

        if "concise_mode" in data:
            personality.concise_mode = bool(
                data["concise_mode"]
            )

        if "custom_attributes" in data:
            personality.custom_attributes = dict(
                data["custom_attributes"]
            )

        return personality

    def reset(self) -> None:
        """Reset personality settings to defaults."""

        default = Personality()

        self.name = default.name
        self.identity = default.identity
        self.traits = list(default.traits)
        self.communication_style = default.communication_style
        self.humor_enabled = default.humor_enabled
        self.empathy_enabled = default.empathy_enabled
        self.concise_mode = default.concise_mode
        self.custom_attributes.clear()


# ====================================================================
# DEFAULT PERSONALITY
# ====================================================================

renix_personality = Personality()


__all__ = [
    "Personality",
    "renix_personality",
]


