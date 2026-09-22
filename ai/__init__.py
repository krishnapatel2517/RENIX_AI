"""
RENIX AI
Artificial Intelligence Package

This package contains the AI layer of RENIX, including:

- Large Language Model providers
- Model routing
- Streaming
- Fallback systems
- Reasoning
- Planning
- Problem solving
- Task decomposition
- Verification
- Criticism
- Autonomous agents
- Personality
- Tone
- Emotions
- Behaviour

The package is designed to provide a unified AI foundation
for the RENIX orchestrator.
"""

from __future__ import annotations


# ============================================================================
# PACKAGE INFORMATION
# ============================================================================

__title__ = "RENIX AI"
__description__ = (
    "Artificial intelligence and autonomous agent layer "
    "for the RENIX assistant."
)
__version__ = "1.0.0"
__author__ = "RENIX"
__license__ = "Proprietary"


# ============================================================================
# PACKAGE FLAGS
# ============================================================================

AI_PACKAGE_AVAILABLE = True


# ============================================================================
# SAFE IMPORT HELPERS
# ============================================================================

def package_info() -> dict[str, str]:
    """
    Return information about the RENIX AI package.
    """

    return {
        "title": __title__,
        "description": __description__,
        "version": __version__,
        "author": __author__,
        "license": __license__,
    }


def is_available() -> bool:
    """
    Return whether the RENIX AI package is available.
    """

    return AI_PACKAGE_AVAILABLE


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "__title__",
    "__description__",
    "__version__",
    "__author__",
    "__license__",
    "AI_PACKAGE_AVAILABLE",
    "package_info",
    "is_available",
]


