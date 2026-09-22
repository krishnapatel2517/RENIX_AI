"""
RENIX Gesture System
====================

Central package for RENIX hand and gesture interaction.

The gesture system converts camera/vision landmarks into high-level
commands that can be consumed by the holographic UI, computer controller,
automation engine, and core orchestrator.

Supported gesture concepts include:

    - Pinch
    - Swipe
    - Grab
    - Rotate
    - Zoom
    - Drag
    - Point
    - Palm
    - Two-hand gestures

The package intentionally keeps gesture recognition separate from
computer-control logic. Gesture modules detect and describe intent;
higher-level systems decide what action should be performed.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core gesture engine
# ---------------------------------------------------------------------------

try:
    from .gesture_engine import (
        GestureEngine,
        GestureEvent,
        GestureState,
    )
except ImportError:
    GestureEngine = None
    GestureEvent = None
    GestureState = None


# ---------------------------------------------------------------------------
# Hand / finger tracking
# ---------------------------------------------------------------------------

try:
    from .hand_tracker import (
        HandTracker,
    )
except ImportError:
    HandTracker = None


try:
    from .finger_tracker import (
        FingerTracker,
    )
except ImportError:
    FingerTracker = None


# ---------------------------------------------------------------------------
# Gesture classification and mapping
# ---------------------------------------------------------------------------

try:
    from .gesture_classifier import (
        GestureClassifier,
    )
except ImportError:
    GestureClassifier = None


try:
    from .gesture_mapper import (
        GestureMapper,
    )
except ImportError:
    GestureMapper = None


# ---------------------------------------------------------------------------
# Individual gestures
# ---------------------------------------------------------------------------

try:
    from .pinch import (
        PinchGesture,
    )
except ImportError:
    PinchGesture = None


try:
    from .swipe import (
        SwipeGesture,
    )
except ImportError:
    SwipeGesture = None


try:
    from .grab import (
        GrabGesture,
    )
except ImportError:
    GrabGesture = None


try:
    from .rotate import (
        RotateGesture,
    )
except ImportError:
    RotateGesture = None


try:
    from .zoom import (
        ZoomGesture,
    )
except ImportError:
    ZoomGesture = None


try:
    from .drag import (
        DragGesture,
    )
except ImportError:
    DragGesture = None


try:
    from .point import (
        PointGesture,
    )
except ImportError:
    PointGesture = None


try:
    from .palm import (
        PalmGesture,
    )
except ImportError:
    PalmGesture = None


try:
    from .two_hand import (
        TwoHandGesture,
    )
except ImportError:
    TwoHandGesture = None


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

__version__ = "1.0.0"
__author__ = "RENIX"
__description__ = (
    "RENIX real-time hand gesture recognition "
    "and interaction system."
)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    # Engine
    "GestureEngine",
    "GestureEvent",
    "GestureState",

    # Tracking
    "HandTracker",
    "FingerTracker",

    # Classification / mapping
    "GestureClassifier",
    "GestureMapper",

    # Gestures
    "PinchGesture",
    "SwipeGesture",
    "GrabGesture",
    "RotateGesture",
    "ZoomGesture",
    "DragGesture",
    "PointGesture",
    "PalmGesture",
    "TwoHandGesture",
]


