"""
RENIX AI
LLM Package

Central package for RENIX Large Language Model functionality.

This package contains:

- LLM providers
- Model routing
- Streaming responses
- Provider fallback
- Model configuration
"""

from __future__ import annotations


# ============================================================================
# PACKAGE INFORMATION
# ============================================================================

__title__ = "RENIX LLM"
__description__ = (
    "Large Language Model infrastructure for RENIX."
)
__version__ = "1.0.0"


# ============================================================================
# PACKAGE STATE
# ============================================================================

LLM_PACKAGE_AVAILABLE = True


# ============================================================================
# PACKAGE INFORMATION FUNCTION
# ============================================================================

def package_info() -> dict[str, str]:
    """
    Return information about the RENIX LLM package.
    """

    return {
        "title": __title__,
        "description": __description__,
        "version": __version__,
    }


# ============================================================================
# AVAILABILITY
# ============================================================================

def is_available() -> bool:
    """
    Return whether the LLM package is available.
    """

    return LLM_PACKAGE_AVAILABLE


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "__title__",
    "__description__",
    "__version__",
    "LLM_PACKAGE_AVAILABLE",
    "package_info",
    "is_available",
]


