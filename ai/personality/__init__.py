"""
RENIX AI Personality Package

Contains RENIX's personality, tone, emotional state,
and behavioral systems.
"""

from .personality import Personality
from .tone import Tone
from .emotions import EmotionState
from .behavior import Behavior


__all__ = [
    "Personality",
    "Tone",
    "EmotionState",
    "Behavior",
]


