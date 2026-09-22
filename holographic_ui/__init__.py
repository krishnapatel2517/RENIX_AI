"""
RENIX Holographic UI
====================

The holographic interface layer of RENIX.

This package is responsible for:
    - Holographic desktop rendering
    - HUD components
    - Spatial UI
    - Gesture-based UI interaction
    - Voice-based UI interaction
    - 3D objects and scenes
    - Particles and visual effects
    - Notifications
    - Commands
    - Files
    - System controls
    - Browser UI
    - Terminal UI
    - Settings UI
    - UI widgets
    - Themes and animations

Architecture
------------

vision/
    Detects what the user is doing.

gestures/
    Converts hand movement into gestures.

voice/
    Converts speech into commands.

core/
    Decides what RENIX should do.

holographic_ui/
    Displays the result and handles visual interaction.

This package intentionally keeps UI logic separate from:
    - AI reasoning
    - memory
    - computer control
    - vision
    - gesture detection
    - voice processing
"""

from __future__ import annotations


# ============================================================================
# PACKAGE METADATA
# ============================================================================

__title__ = "RENIX Holographic UI"
__description__ = (
    "Holographic user interface system for RENIX AI"
)
__version__ = "1.0.0"
__author__ = "RENIX"
__license__ = "MIT"


# ============================================================================
# OPTIONAL COMPONENT EXPORTS
# ============================================================================
#
# Components are imported lazily where possible.
# This prevents importing the complete graphical stack simply by doing:
#
#     import RENIX.holographic_ui
#
# This is important because some UI components may depend on optional
# rendering libraries.
# ============================================================================


def get_app():
    """
    Return the RENIX holographic application class.

    Importing lazily prevents optional UI dependencies from being loaded
    during package discovery.
    """

    from .app import RENIXApp

    return RENIXApp


def get_window_manager():
    """
    Return the holographic window manager class.
    """

    from .window_manager import WindowManager

    return WindowManager


def get_scene_manager():
    """
    Return the scene manager class.
    """

    from .scene_manager import SceneManager

    return SceneManager


def get_spatial_manager():
    """
    Return the spatial UI manager class.
    """

    from .spatial_manager import SpatialManager

    return SpatialManager


def get_gesture_interface():
    """
    Return the gesture-to-UI interface.
    """

    from .gesture_interface import GestureInterface

    return GestureInterface


def get_voice_interface():
    """
    Return the voice-to-UI interface.
    """

    from .voice_interface import VoiceInterface

    return VoiceInterface


def get_hologram_engine():
    """
    Return the main hologram rendering engine.
    """

    from .hologram_engine import HologramEngine

    return HologramEngine


def get_hud_engine():
    """
    Return the HUD engine.
    """

    from .hud_engine import HUDEngine

    return HUDEngine


def get_animation_engine():
    """
    Return the animation engine.
    """

    from .animation_engine import AnimationEngine

    return AnimationEngine


def get_particle_engine():
    """
    Return the particle engine.
    """

    from .particle_engine import ParticleEngine

    return ParticleEngine


def get_3d_engine():
    """
    Return the 3D rendering engine.
    """

    try:
        from .engine_3d import Engine3D
        return Engine3D
    except Exception:
        import importlib
        module = importlib.import_module(".3d_engine", __name__)
        return module.Engine3D


# ============================================================================
# SAFE OPTIONAL IMPORT HELPER
# ============================================================================


def is_available() -> bool:
    """
    Check whether the holographic UI package can be imported.

    This does not start the UI.

    Returns
    -------
    bool
        True if the package itself is available.
    """

    return True


# ============================================================================
# PACKAGE EXPORTS
# ============================================================================

__all__ = [
    "__title__",
    "__description__",
    "__version__",
    "__author__",
    "__license__",
    "get_app",
    "get_window_manager",
    "get_scene_manager",
    "get_spatial_manager",
    "get_gesture_interface",
    "get_voice_interface",
    "get_hologram_engine",
    "get_hud_engine",
    "get_animation_engine",
    "get_particle_engine",
    "get_3d_engine",
    "is_available",
]


