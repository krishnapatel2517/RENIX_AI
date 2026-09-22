"""
RENIX Vision System
===================

Central package for RENIX's computer-vision layer.

The vision system provides the foundation for:

- Camera input
- Camera management
- Face detection
- Face recognition
- Person detection
- Object detection
- Object tracking
- Hand detection
- Pose detection
- Scene understanding
- OCR
- Document detection
- QR detection
- Screen capture
- Screen understanding
- Unified vision pipelines

Architecture
------------

Camera
   ↓
Camera Manager
   ↓
Vision Pipeline
   ├── Face Detection
   ├── Face Recognition
   ├── Person Detection
   ├── Object Detection
   ├── Object Tracking
   ├── Hand Detection
   ├── Pose Detection
   ├── Scene Understanding
   ├── OCR
   ├── Document Detection
   └── QR Detection

The package intentionally keeps imports lightweight.
Individual modules are loaded lazily through helper functions
to avoid unnecessary initialization of camera/vision libraries.
"""

from __future__ import annotations


# ============================================================================
# PACKAGE INFORMATION
# ============================================================================

__version__ = "1.0.0"
__author__ = "RENIX"
__project__ = "RENIX AI"
__package_name__ = "renix.vision"


# ============================================================================
# FEATURE FLAGS
# ============================================================================

VISION_FEATURES = {
    "camera": True,
    "camera_manager": True,
    "face_detection": True,
    "face_recognition": True,
    "person_detection": True,
    "object_detection": True,
    "object_tracking": True,
    "hand_detection": True,
    "pose_detection": True,
    "scene_understanding": True,
    "ocr": True,
    "document_detection": True,
    "qr_detection": True,
    "screen_capture": True,
    "screen_understanding": True,
    "vision_pipeline": True,
}


# ============================================================================
# LAZY IMPORT HELPERS
# ============================================================================

def get_camera():
    """Return the Camera class."""
    from .camera import Camera

    return Camera


def get_camera_manager():
    """Return the CameraManager class."""
    from .camera_manager import CameraManager

    return CameraManager


def get_face_detector():
    """Return the face detector implementation."""
    from .face_detection import FaceDetector

    return FaceDetector


def get_face_recognizer():
    """Return the face recognition implementation."""
    from .face_recognition import FaceRecognizer

    return FaceRecognizer


def get_person_detector():
    """Return the person detector implementation."""
    from .person_detection import PersonDetector

    return PersonDetector


def get_object_detector():
    """Return the object detector implementation."""
    from .object_detection import ObjectDetector

    return ObjectDetector


def get_object_tracker():
    """Return the object tracker implementation."""
    from .object_tracking import ObjectTracker

    return ObjectTracker


def get_hand_detector():
    """Return the hand detector implementation."""
    from .hand_detection import HandDetector

    return HandDetector


def get_pose_detector():
    """Return the pose detector implementation."""
    from .pose_detection import PoseDetector

    return PoseDetector


def get_scene_understanding():
    """Return the scene-understanding implementation."""
    from .scene_understanding import SceneUnderstanding

    return SceneUnderstanding


def get_ocr_engine():
    """Return the OCR engine."""
    from .ocr import OCREngine

    return OCREngine


def get_document_detector():
    """Return the document detector."""
    from .document_detection import DocumentDetector

    return DocumentDetector


def get_qr_detector():
    """Return the QR detector."""
    from .qr_detection import QRDetector

    return QRDetector


def get_screen_capture():
    """Return the screen-capture implementation."""
    from .screen_capture import ScreenCapture

    return ScreenCapture


def get_screen_understanding():
    """Return the screen-understanding implementation."""
    from .screen_understanding import ScreenUnderstanding

    return ScreenUnderstanding


def get_vision_pipeline():
    """Return the unified VisionPipeline."""
    from .vision_pipeline import VisionPipeline

    return VisionPipeline


# ============================================================================
# FEATURE CHECKING
# ============================================================================

def is_feature_enabled(
    feature: str,
) -> bool:
    """
    Check whether a vision feature is enabled.

    Parameters
    ----------
    feature:
        Feature name such as ``face_detection`` or
        ``hand_detection``.
    """

    return bool(
        VISION_FEATURES.get(
            feature,
            False,
        )
    )


def enabled_features() -> list[str]:
    """Return all enabled vision features."""

    return [
        name
        for name, enabled
        in VISION_FEATURES.items()
        if enabled
    ]


# ============================================================================
# PACKAGE HEALTH
# ============================================================================

def get_package_info() -> dict:
    """
    Return basic information about the RENIX vision package.
    """

    return {
        "package": __package_name__,
        "version": __version__,
        "author": __author__,
        "project": __project__,
        "features": dict(VISION_FEATURES),
        "enabled_features": enabled_features(),
    }


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "__version__",
    "__author__",
    "__project__",
    "__package_name__",
    "VISION_FEATURES",

    "get_camera",
    "get_camera_manager",

    "get_face_detector",
    "get_face_recognizer",

    "get_person_detector",

    "get_object_detector",
    "get_object_tracker",

    "get_hand_detector",
    "get_pose_detector",

    "get_scene_understanding",

    "get_ocr_engine",
    "get_document_detector",
    "get_qr_detector",

    "get_screen_capture",
    "get_screen_understanding",

    "get_vision_pipeline",

    "is_feature_enabled",
    "enabled_features",
    "get_package_info",
]


