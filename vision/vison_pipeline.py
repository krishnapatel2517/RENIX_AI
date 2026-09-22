"""
RENIX Vision Pipeline

Central vision-processing pipeline for RENIX.

Responsibilities:
    Camera/screen frame -> preprocessing -> detection -> tracking
    -> OCR -> scene understanding -> structured VisionResult.

This module coordinates vision components. It does not directly
control the computer, mouse, keyboard, files, or applications.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None


logger = logging.getLogger("RENIX.vision.vision_pipeline")


# ============================================================================
# OPTIONAL RENIX COMPONENT IMPORTS
# ============================================================================

try:
    from .face_detection import FaceDetection
except ImportError:
    FaceDetection = None

try:
    from .face_recognition import FaceRecognition
except ImportError:
    FaceRecognition = None

try:
    from .person_detection import PersonDetection
except ImportError:
    PersonDetection = None

try:
    from .object_detection import ObjectDetection
except ImportError:
    ObjectDetection = None

try:
    from .object_tracking import ObjectTracking
except ImportError:
    ObjectTracking = None

try:
    from .hand_detection import HandDetection
except ImportError:
    HandDetection = None

try:
    from .pose_detection import PoseDetection
except ImportError:
    PoseDetection = None

try:
    from .scene_understanding import SceneUnderstanding
except ImportError:
    SceneUnderstanding = None

try:
    from .ocr import OCR
except ImportError:
    OCR = None

try:
    from .document_detection import DocumentDetection
except ImportError:
    DocumentDetection = None

try:
    from .qr_detection import QRDetection
except ImportError:
    QRDetection = None

try:
    from .screen_understanding import (
        ScreenUnderstanding,
        ScreenAnalysis,
    )
except ImportError:
    ScreenUnderstanding = None
    ScreenAnalysis = None


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class VisionPipelineConfig:
    """Configuration controlling which vision stages are enabled."""

    enabled: bool = True

    enable_face_detection: bool = True
    enable_face_recognition: bool = False
    enable_person_detection: bool = True
    enable_object_detection: bool = True
    enable_object_tracking: bool = True
    enable_hand_detection: bool = True
    enable_pose_detection: bool = True
    enable_scene_understanding: bool = True
    enable_ocr: bool = True
    enable_document_detection: bool = True
    enable_qr_detection: bool = True
    enable_screen_understanding: bool = True

    draw_debug_overlay: bool = False

    max_frame_width: int = 1920

    target_fps: float = 30.0

    confidence_threshold: float = 0.35

    skip_duplicate_frames: bool = False

    duplicate_frame_threshold: float = 0.995

    keep_last_result: bool = True


@dataclass
class Detection:
    """Generic normalized detection."""

    detection_type: str

    label: str

    confidence: float

    box: Optional[dict[str, float]] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_type": self.detection_type,
            "label": self.label,
            "confidence": self.confidence,
            "box": self.box,
            "metadata": dict(self.metadata),
        }


@dataclass
class VisionResult:
    """Complete output of one vision pipeline pass."""

    timestamp: float

    frame_number: int

    width: int

    height: int

    processing_time_ms: float

    fps: float

    success: bool = True

    duplicate_frame: bool = False

    faces: list[Any] = field(default_factory=list)

    recognized_faces: list[Any] = field(
        default_factory=list
    )

    persons: list[Any] = field(default_factory=list)

    objects: list[Any] = field(default_factory=list)

    tracked_objects: list[Any] = field(
        default_factory=list
    )

    hands: list[Any] = field(default_factory=list)

    poses: list[Any] = field(default_factory=list)

    scene: Any = None

    text: list[Any] = field(default_factory=list)

    documents: list[Any] = field(
        default_factory=list
    )

    qr_codes: list[Any] = field(
        default_factory=list
    )

    screen: Any = None

    detections: list[Detection] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "width": self.width,
            "height": self.height,
            "processing_time_ms": (
                self.processing_time_ms
            ),
            "fps": self.fps,
            "success": self.success,
            "duplicate_frame": (
                self.duplicate_frame
            ),
            "faces": self.faces,
            "recognized_faces": (
                self.recognized_faces
            ),
            "persons": self.persons,
            "objects": self.objects,
            "tracked_objects": (
                self.tracked_objects
            ),
            "hands": self.hands,
            "poses": self.poses,
            "scene": self.scene,
            "text": self.text,
            "documents": self.documents,
            "qr_codes": self.qr_codes,
            "screen": self.screen,
            "detections": [
                item.to_dict()
                for item in self.detections
            ],
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }


# ============================================================================
# VISION PIPELINE
# ============================================================================


class VisionPipeline:
    """
    Main RENIX vision orchestration layer.

    The pipeline is deliberately modular. Each detector is optional, so
    RENIX can still start when an individual computer-vision dependency
    or model is unavailable.
    """

    def __init__(
        self,
        config: Optional[
            VisionPipelineConfig
        ] = None,
        *,
        components: Optional[
            dict[str, Any]
        ] = None,
    ) -> None:

        self.config = (
            config
            or VisionPipelineConfig()
        )

        self._lock = threading.RLock()

        self._frame_number = 0

        self._last_result: Optional[
            VisionResult
        ] = None

        self._last_frame: Any = None

        self._last_frame_time: Optional[
            float
        ] = None

        self._fps_history: list[float] = []

        self._callbacks: list[
            Callable[[VisionResult], None]
        ] = []

        self._components: dict[
            str, Any
        ] = {}

        if components:

            self._components.update(
                components
            )

        self._initialize_components()

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def _initialize_components(
        self,
    ) -> None:

        self._create_component(
            "face_detection",
            FaceDetection,
            self.config.enable_face_detection,
        )

        self._create_component(
            "face_recognition",
            FaceRecognition,
            self.config.enable_face_recognition,
        )

        self._create_component(
            "person_detection",
            PersonDetection,
            self.config.enable_person_detection,
        )

        self._create_component(
            "object_detection",
            ObjectDetection,
            self.config.enable_object_detection,
        )

        self._create_component(
            "object_tracking",
            ObjectTracking,
            self.config.enable_object_tracking,
        )

        self._create_component(
            "hand_detection",
            HandDetection,
            self.config.enable_hand_detection,
        )

        self._create_component(
            "pose_detection",
            PoseDetection,
            self.config.enable_pose_detection,
        )

        self._create_component(
            "scene_understanding",
            SceneUnderstanding,
            self.config.enable_scene_understanding,
        )

        self._create_component(
            "ocr",
            OCR,
            self.config.enable_ocr,
        )

        self._create_component(
            "document_detection",
            DocumentDetection,
            self.config.enable_document_detection,
        )

        self._create_component(
            "qr_detection",
            QRDetection,
            self.config.enable_qr_detection,
        )

        self._create_component(
            "screen_understanding",
            ScreenUnderstanding,
            self.config.enable_screen_understanding,
        )

    def _create_component(
        self,
        name: str,
        component_class: Any,
        enabled: bool,
    ) -> None:

        if name in self._components:
            return

        if not enabled:
            return

        if component_class is None:
            return

        try:
            self._components[name] = (
                component_class()
            )

        except TypeError:
            try:
                self._components[name] = (
                    component_class
                )
            except Exception:
                logger.debug(
                    "Unable to initialize %s.",
                    name,
                    exc_info=True,
                )

        except Exception:
            logger.exception(
                "Failed to initialize vision "
                "component: %s",
                name,
            )

    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================

    def process(
        self,
        frame: Any,
        *,
        timestamp: Optional[float] = None,
        frame_number: Optional[int] = None,
        source: str = "camera",
    ) -> VisionResult:

        if not self.config.enabled:

            return self._disabled_result(
                frame,
                timestamp=timestamp,
                frame_number=frame_number,
            )

        if frame is None:

            raise ValueError(
                "VisionPipeline.process() "
                "received None."
            )

        if np is None:

            raise RuntimeError(
                "NumPy is required by RENIX "
                "VisionPipeline."
            )

        start = time.perf_counter()

        image = np.asarray(frame)

        if image.size == 0:

            raise ValueError(
                "VisionPipeline received an "
                "empty frame."
            )

        image = self._prepare_frame(
            image
        )

        height, width = image.shape[:2]

        current_time = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        with self._lock:

            if frame_number is None:

                self._frame_number += 1

                current_frame_number = (
                    self._frame_number
                )

            else:

                current_frame_number = (
                    frame_number
                )

                self._frame_number = max(
                    self._frame_number,
                    frame_number,
                )

        duplicate = False

        if self.config.skip_duplicate_frames:

            duplicate = (
                self._is_duplicate_frame(
                    image
                )
            )

        result = VisionResult(
            timestamp=current_time,
            frame_number=current_frame_number,
            width=int(width),
            height=int(height),
            processing_time_ms=0.0,
            fps=self._calculate_fps(
                current_time
            ),
            duplicate_frame=duplicate,
        )

        if duplicate:

            if self._last_result is not None:

                result = self._clone_result(
                    self._last_result,
                    timestamp=current_time,
                    frame_number=(
                        current_frame_number
                    ),
                )

                result.duplicate_frame = True

            result.processing_time_ms = (
                (
                    time.perf_counter()
                    - start
                )
                * 1000
            )

            self._store_frame(
                image,
                current_time,
            )

            self._publish_result(
                result
            )

            return result

        # --------------------------------------------------------------------
        # 1. Face detection
        # --------------------------------------------------------------------

        self._run_stage(
            result,
            "faces",
            self._run_face_detection,
            image,
        )

        # --------------------------------------------------------------------
        # 2. Face recognition
        # --------------------------------------------------------------------

        if result.faces:

            self._run_stage(
                result,
                "recognized_faces",
                self._run_face_recognition,
                image,
                result.faces,
            )

        # --------------------------------------------------------------------
        # 3. Person detection
        # --------------------------------------------------------------------

        self._run_stage(
            result,
            "persons",
            self._run_person_detection,
            image,
        )

        # --------------------------------------------------------------------
        # 4. Object detection
        # --------------------------------------------------------------------

        self._run_stage(
            result,
            "objects",
            self._run_object_detection,
            image,
        )

        # --------------------------------------------------------------------
        # 5. Object tracking
        # --------------------------------------------------------------------

        if result.objects:

            self._run_stage(
                result,
                "tracked_objects",
                self._run_object_tracking,
                image,
                result.objects,
            )

        # --------------------------------------------------------------------
        # 6. Hand detection
        # --------------------------------------------------------------------

        self._run_stage(
            result,
            "hands",
            self._run_hand_detection,
            image,
        )

        # --------------------------------------------------------------------
        # 7. Pose detection
        # --------------------------------------------------------------------

        self._run_stage(
            result,
            "poses",
            self._run_pose_detection,
            image,
        )

        # --------------------------------------------------------------------
        # 8. OCR
        # --------------------------------------------------------------------

        self._run_stage(
            result,
            "text",
            self._run_ocr,
            image,
        )

        # --------------------------------------------------------------------
        # 9. Document detection
        # --------------------------------------------------------------------

        self._run_stage(
            result,
            "documents",
            self._run_document_detection,
            image,
        )

        # --------------------------------------------------------------------
        # 10. QR detection
        # --------------------------------------------------------------------

        self._run_stage(
            result,
            "qr_codes",
            self._run_qr_detection,
            image,
        )

        # --------------------------------------------------------------------
        # 11. Scene understanding
        # --------------------------------------------------------------------

        self._run_stage(
            result,
            "scene",
            self._run_scene_understanding,
            image,
            result,
        )

        # --------------------------------------------------------------------
        # 12. Screen understanding
        # --------------------------------------------------------------------

        if (
            source.lower()
            in {
                "screen",
                "desktop",
                "window",
            }
            or self.config.enable_screen_understanding
        ):

            self._run_stage(
                result,
                "screen",
                self._run_screen_understanding,
                image,
            )

        # --------------------------------------------------------------------
        # Normalize detections
        # --------------------------------------------------------------------

        self._build_generic_detections(
            result
        )

        result.processing_time_ms = (
            (
                time.perf_counter()
                - start
            )
            * 1000
        )

        result.fps = self._calculate_fps(
            current_time
        )

        result.metadata.update(
            {
                "source": source,
                "component_count": len(
                    self._components
                ),
                "enabled_stages": (
                    self.get_enabled_stages()
                ),
            }
        )

        self._store_frame(
            image,
            current_time,
        )

        self._store_result(
            result
        )

        self._publish_result(
            result
        )

        return result

    # ========================================================================
    # FRAME PREPARATION
    # ========================================================================

    def _prepare_frame(
        self,
        frame: Any,
    ) -> Any:

        if cv2 is None:

            return frame

        height, width = frame.shape[:2]

        max_width = (
            self.config.max_frame_width
        )

        if (
            max_width > 0
            and width > max_width
        ):

            scale = (
                max_width / width
            )

            frame = cv2.resize(
                frame,
                (
                    max_width,
                    max(
                        1,
                        int(
                            height * scale
                        ),
                    ),
                ),
                interpolation=(
                    cv2.INTER_AREA
                ),
            )

        return frame

    # ========================================================================
    # STAGES
    # ========================================================================

    def _run_face_detection(
        self,
        image: Any,
    ) -> Any:

        component = self._components.get(
            "face_detection"
        )

        if component is None:
            return []

        return self._call_component(
            component,
            [
                "detect",
                "process",
                "analyze",
            ],
            image,
        )

    def _run_face_recognition(
        self,
        image: Any,
        faces: Any,
    ) -> Any:

        component = self._components.get(
            "face_recognition"
        )

        if component is None:
            return []

        return self._call_component(
            component,
            [
                "recognize",
                "process",
                "analyze",
            ],
            image,
            faces,
        )

    def _run_person_detection(
        self,
        image: Any,
    ) -> Any:

        component = self._components.get(
            "person_detection"
        )

        if component is None:
            return []

        return self._call_component(
            component,
            [
                "detect",
                "process",
                "analyze",
            ],
            image,
        )

    def _run_object_detection(
        self,
        image: Any,
    ) -> Any:

        component = self._components.get(
            "object_detection"
        )

        if component is None:
            return []

        return self._call_component(
            component,
            [
                "detect",
                "process",
                "analyze",
            ],
            image,
        )

    def _run_object_tracking(
        self,
        image: Any,
        objects: Any,
    ) -> Any:

        component = self._components.get(
            "object_tracking"
        )

        if component is None:
            return objects

        return self._call_component(
            component,
            [
                "track",
                "update",
                "process",
                "analyze",
            ],
            image,
            objects,
        )

    def _run_hand_detection(
        self,
        image: Any,
    ) -> Any:

        component = self._components.get(
            "hand_detection"
        )

        if component is None:
            return []

        return self._call_component(
            component,
            [
                "detect",
                "process",
                "analyze",
            ],
            image,
        )

    def _run_pose_detection(
        self,
        image: Any,
    ) -> Any:

        component = self._components.get(
            "pose_detection"
        )

        if component is None:
            return []

        return self._call_component(
            component,
            [
                "detect",
                "process",
                "analyze",
            ],
            image,
        )

    def _run_scene_understanding(
        self,
        image: Any,
        result: VisionResult,
    ) -> Any:

        component = self._components.get(
            "scene_understanding"
        )

        if component is None:
            return None

        return self._call_component(
            component,
            [
                "understand",
                "analyze",
                "process",
            ],
            image,
            result,
        )

    def _run_ocr(
        self,
        image: Any,
    ) -> Any:

        component = self._components.get(
            "ocr"
        )

        if component is None:

            # ScreenUnderstanding can serve
            # as a fallback OCR provider.
            screen_engine = (
                self._components.get(
                    "screen_understanding"
                )
            )

            if screen_engine is not None:

                method = getattr(
                    screen_engine,
                    "extract_text",
                    None,
                )

                if callable(method):

                    return method(image)

            return []

        return self._call_component(
            component,
            [
                "extract",
                "extract_text",
                "recognize",
                "process",
                "analyze",
            ],
            image,
        )

    def _run_document_detection(
        self,
        image: Any,
    ) -> Any:

        component = self._components.get(
            "document_detection"
        )

        if component is None:
            return []

        return self._call_component(
            component,
            [
                "detect",
                "process",
                "analyze",
            ],
            image,
        )

    def _run_qr_detection(
        self,
        image: Any,
    ) -> Any:

        component = self._components.get(
            "qr_detection"
        )

        if component is None:
            return []

        return self._call_component(
            component,
            [
                "detect",
                "scan",
                "process",
                "analyze",
            ],
            image,
        )

    def _run_screen_understanding(
        self,
        image: Any,
    ) -> Any:

        component = self._components.get(
            "screen_understanding"
        )

        if component is None:
            return None

        return self._call_component(
            component,
            [
                "analyze",
                "understand",
                "process",
            ],
            image,
        )

    # ========================================================================
    # COMPONENT INVOCATION
    # ========================================================================

    def _call_component(
        self,
        component: Any,
        method_names: list[str],
        *args: Any,
    ) -> Any:

        for method_name in method_names:

            method = getattr(
                component,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:

                return method(*args)

            except TypeError:

                # Some RENIX components may
                # accept fewer arguments.
                try:

                    return method(
                        args[0]
                    )

                except Exception:

                    logger.debug(
                        "Component %s failed "
                        "with fallback signature.",
                        type(component).__name__,
                        exc_info=True,
                    )

            except Exception:

                logger.exception(
                    "Vision component %s "
                    "failed in %s().",
                    type(component).__name__,
                    method_name,
                )

                return []

        return []

    def _run_stage(
        self,
        result: VisionResult,
        field_name: str,
        function: Callable,
        *args: Any,
    ) -> None:

        try:

            value = function(*args)

            setattr(
                result,
                field_name,
                self._safe_result_value(
                    value
                ),
            )

        except Exception as exc:

            message = (
                f"{field_name}: {exc}"
            )

            result.errors.append(
                message
            )

            logger.exception(
                "Vision stage failed: %s",
                field_name,
            )

    # ========================================================================
    # RESULT NORMALIZATION
    # ========================================================================

    def _safe_result_value(
        self,
        value: Any,
    ) -> Any:

        if value is None:
            return None

        if isinstance(
            value,
            tuple,
        ):

            return list(value)

        return value

    def _build_generic_detections(
        self,
        result: VisionResult,
    ) -> None:

        result.detections.clear()

        self._append_generic_detections(
            result,
            result.faces,
            "face",
        )

        self._append_generic_detections(
            result,
            result.persons,
            "person",
        )

        self._append_generic_detections(
            result,
            result.objects,
            "object",
        )

        self._append_generic_detections(
            result,
            result.hands,
            "hand",
        )

        self._append_generic_detections(
            result,
            result.qr_codes,
            "qr",
        )

    def _append_generic_detections(
        self,
        result: VisionResult,
        values: Any,
        detection_type: str,
    ) -> None:

        if values is None:
            return

        if not isinstance(
            values,
            (list, tuple),
        ):

            values = [values]

        for item in values:

            detection = (
                self._normalize_detection(
                    item,
                    detection_type,
                )
            )

            if detection is not None:

                if (
                    detection.confidence
                    >= self.config.confidence_threshold
                ):

                    result.detections.append(
                        detection
                    )

    def _normalize_detection(
        self,
        item: Any,
        detection_type: str,
    ) -> Optional[Detection]:

        if item is None:
            return None

        if isinstance(
            item,
            Detection,
        ):

            return item

        if isinstance(
            item,
            dict,
        ):

            label = str(
                item.get(
                    "label",
                    item.get(
                        "name",
                        detection_type,
                    ),
                )
            )

            confidence = self._to_float(
                item.get(
                    "confidence",
                    item.get(
                        "score",
                        1.0,
                    ),
                ),
                1.0,
            )

            box = item.get(
                "box",
                item.get(
                    "bbox"
                ),
            )

            return Detection(
                detection_type=detection_type,
                label=label,
                confidence=confidence,
                box=self._normalize_box(
                    box
                ),
                metadata=dict(item),
            )

        label = getattr(
            item,
            "label",
            getattr(
                item,
                "name",
                detection_type,
            ),
        )

        confidence = getattr(
            item,
            "confidence",
            getattr(
                item,
                "score",
                1.0,
            ),
        )

        box = getattr(
            item,
            "box",
            getattr(
                item,
                "bbox",
                None,
            ),
        )

        return Detection(
            detection_type=detection_type,
            label=str(label),
            confidence=self._to_float(
                confidence,
                1.0,
            ),
            box=self._normalize_box(
                box
            ),
        )

    def _normalize_box(
        self,
        box: Any,
    ) -> Optional[
        dict[str, float]
    ]:

        if box is None:
            return None

        if isinstance(
            box,
            dict,
        ):

            keys = {
                "x",
                "y",
                "width",
                "height",
            }

            if keys.issubset(
                box.keys()
            ):

                return {
                    key: float(
                        box[key]
                    )
                    for key in keys
                }

            if {
                "x1",
                "y1",
                "x2",
                "y2",
            }.issubset(
                box.keys()
            ):

                return {
                    "x": float(
                        box["x1"]
                    ),
                    "y": float(
                        box["y1"]
                    ),
                    "width": float(
                        box["x2"]
                        - box["x1"]
                    ),
                    "height": float(
                        box["y2"]
                        - box["y1"]
                    ),
                }

        if isinstance(
            box,
            (list, tuple),
        ) and len(box) >= 4:

            return {
                "x": float(box[0]),
                "y": float(box[1]),
                "width": float(box[2]),
                "height": float(box[3]),
            }

        return None

    def _to_float(
        self,
        value: Any,
        default: float,
    ) -> float:

        try:

            return float(value)

        except (
            TypeError,
            ValueError,
        ):

            return default

    # ========================================================================
    # DUPLICATE DETECTION
    # ========================================================================

    def _is_duplicate_frame(
        self,
        image: Any,
    ) -> bool:

        with self._lock:

            previous = self._last_frame

        if previous is None:
            return False

        if previous.shape != image.shape:
            return False

        try:

            if cv2 is not None:

                a = self._small_gray(
                    previous
                )

                b = self._small_gray(
                    image
                )

                diff = cv2.absdiff(
                    a,
                    b,
                )

                similarity = (
                    1.0
                    - (
                        float(
                            np.mean(diff)
                        )
                        / 255.0
                    )
                )

            else:

                a = previous.astype(
                    np.float32
                )

                b = image.astype(
                    np.float32
                )

                diff = np.mean(
                    np.abs(a - b)
                )

                similarity = (
                    1.0
                    - float(diff)
                    / 255.0
                )

            return (
                similarity
                >= self.config.duplicate_frame_threshold
            )

        except Exception:

            return False

    def _small_gray(
        self,
        image: Any,
    ) -> Any:

        if cv2 is None:

            return image

        gray = image

        if image.ndim == 3:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        return cv2.resize(
            gray,
            (
                64,
                64,
            ),
            interpolation=cv2.INTER_AREA,
        )

    # ========================================================================
    # FPS
    # ========================================================================

    def _calculate_fps(
        self,
        timestamp: float,
    ) -> float:

        with self._lock:

            previous = (
                self._last_frame_time
            )

            self._last_frame_time = (
                timestamp
            )

        if previous is None:
            return 0.0

        delta = timestamp - previous

        if delta <= 0:
            return 0.0

        fps = 1.0 / delta

        with self._lock:

            self._fps_history.append(
                fps
            )

            if len(
                self._fps_history
            ) > 30:

                self._fps_history.pop(
                    0
                )

            return (
                sum(
                    self._fps_history
                )
                / len(
                    self._fps_history
                )
            )

    # ========================================================================
    # STORAGE
    # ========================================================================

    def _store_frame(
        self,
        image: Any,
        timestamp: float,
    ) -> None:

        with self._lock:

            self._last_frame = (
                image.copy()
                if hasattr(
                    image,
                    "copy",
                )
                else image
            )

            self._last_frame_time = (
                timestamp
            )

    def _store_result(
        self,
        result: VisionResult,
    ) -> None:

        if not self.config.keep_last_result:
            return

        with self._lock:

            self._last_result = result

    # ========================================================================
    # RESULT CLONING
    # ========================================================================

    def _clone_result(
        self,
        result: VisionResult,
        *,
        timestamp: float,
        frame_number: int,
    ) -> VisionResult:

        return VisionResult(
            timestamp=timestamp,
            frame_number=frame_number,
            width=result.width,
            height=result.height,
            processing_time_ms=(
                result.processing_time_ms
            ),
            fps=result.fps,
            success=result.success,
            duplicate_frame=True,
            faces=list(result.faces),
            recognized_faces=list(
                result.recognized_faces
            ),
            persons=list(result.persons),
            objects=list(result.objects),
            tracked_objects=list(
                result.tracked_objects
            ),
            hands=list(result.hands),
            poses=list(result.poses),
            scene=result.scene,
            text=list(result.text),
            documents=list(result.documents),
            qr_codes=list(result.qr_codes),
            screen=result.screen,
            detections=list(
                result.detections
            ),
            errors=list(result.errors),
            metadata=dict(result.metadata),
        )

    def _disabled_result(
        self,
        frame: Any,
        *,
        timestamp: Optional[float],
        frame_number: Optional[int],
    ) -> VisionResult:

        current_time = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        if np is not None:

            image = np.asarray(frame)

            if image.ndim >= 2:

                height, width = (
                    image.shape[:2]
                )

            else:

                height = 0
                width = 0

        else:

            height = 0
            width = 0

        return VisionResult(
            timestamp=current_time,
            frame_number=(
                frame_number
                if frame_number is not None
                else self._frame_number
            ),
            width=int(width),
            height=int(height),
            processing_time_ms=0.0,
            fps=0.0,
            success=False,
            errors=[
                "Vision pipeline disabled."
            ],
        )

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def add_callback(
        self,
        callback: Callable[
            [VisionResult],
            None,
        ],
    ) -> None:

        if not callable(callback):

            raise TypeError(
                "callback must be callable."
            )

        with self._lock:

            if callback not in self._callbacks:

                self._callbacks.append(
                    callback
                )

    def remove_callback(
        self,
        callback: Callable[
            [VisionResult],
            None,
        ],
    ) -> None:

        with self._lock:

            if callback in self._callbacks:

                self._callbacks.remove(
                    callback
                )

    def _publish_result(
        self,
        result: VisionResult,
    ) -> None:

        with self._lock:

            callbacks = list(
                self._callbacks
            )

        for callback in callbacks:

            try:

                callback(result)

            except Exception:

                logger.exception(
                    "Vision callback failed."
                )

    # ========================================================================
    # COMPONENT MANAGEMENT
    # ========================================================================

    def register_component(
        self,
        name: str,
        component: Any,
    ) -> None:

        if not name:

            raise ValueError(
                "Component name cannot be empty."
            )

        if component is None:

            raise ValueError(
                "Component cannot be None."
            )

        with self._lock:

            self._components[
                name
            ] = component

    def unregister_component(
        self,
        name: str,
    ) -> Optional[Any]:

        with self._lock:

            return self._components.pop(
                name,
                None,
            )

    def get_component(
        self,
        name: str,
    ) -> Optional[Any]:

        with self._lock:

            return self._components.get(
                name
            )

    def get_enabled_stages(
        self,
    ) -> list[str]:

        with self._lock:

            return list(
                self._components.keys()
            )

    # ========================================================================
    # LAST STATE
    # ========================================================================

    def get_last_result(
        self,
    ) -> Optional[VisionResult]:

        with self._lock:

            return self._last_result

    def get_last_frame(
        self,
    ) -> Any:

        with self._lock:

            if self._last_frame is None:
                return None

            if hasattr(
                self._last_frame,
                "copy",
            ):

                return self._last_frame.copy()

            return self._last_frame

    # ========================================================================
    # RESET / SHUTDOWN
    # ========================================================================

    def reset(
        self,
    ) -> None:

        with self._lock:

            self._frame_number = 0

            self._last_result = None

            self._last_frame = None

            self._last_frame_time = None

            self._fps_history.clear()

    def close(
        self,
    ) -> None:

        with self._lock:

            components = list(
                self._components.values()
            )

        for component in components:

            for method_name in (
                "close",
                "shutdown",
                "release",
            ):

                method = getattr(
                    component,
                    method_name,
                    None,
                )

                if not callable(method):
                    continue

                try:

                    method()

                except Exception:

                    logger.debug(
                        "Component cleanup failed.",
                        exc_info=True,
                    )

                break

        self.reset()


# ============================================================================
# DEFAULT PIPELINE
# ============================================================================


_default_pipeline: Optional[
    VisionPipeline
] = None

_default_pipeline_lock = (
    threading.RLock()
)


def get_vision_pipeline(
) -> VisionPipeline:

    global _default_pipeline

    with _default_pipeline_lock:

        if _default_pipeline is None:

            _default_pipeline = (
                VisionPipeline()
            )

        return _default_pipeline


# ============================================================================
# CONVENIENCE API
# ============================================================================


def process_frame(
    frame: Any,
    **kwargs: Any,
) -> VisionResult:

    return (
        get_vision_pipeline()
        .process(
            frame,
            **kwargs,
        )
    )


def register_vision_component(
    name: str,
    component: Any,
) -> None:

    get_vision_pipeline().register_component(
        name,
        component,
    )


def get_last_vision_result(
) -> Optional[VisionResult]:

    return (
        get_vision_pipeline()
        .get_last_result()
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "VisionPipelineConfig",
    "Detection",
    "VisionResult",
    "VisionPipeline",
    "get_vision_pipeline",
    "process_frame",
    "register_vision_component",
    "get_last_vision_result",
]


