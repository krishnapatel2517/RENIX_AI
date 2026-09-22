"""
RENIX Vision — Hand Detection

Responsibilities:
- Detect hands in camera frames.
- Represent hand bounding boxes.
- Represent left/right hand information when available.
- Represent hand landmarks.
- Calculate landmark coordinates.
- Calculate palm center and hand geometry.
- Calculate finger-tip positions.
- Provide a backend abstraction for MediaPipe / YOLO / custom models.
- Provide graceful fallback when no hand-detection backend is configured.

This module detects hands only.

Gesture interpretation belongs to:
    RENIX/gestures/

Face detection belongs to:
    RENIX/vision/face_detection.py

Object detection belongs to:
    RENIX/vision/object_detection.py
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

from .object_detection import BoundingBox


logger = logging.getLogger(
    "RENIX.vision.hand_detection"
)


# ============================================================================
# CONSTANTS
# ============================================================================


# MediaPipe-style 21 landmark indices.
WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


FINGERTIP_INDICES = {
    "thumb": THUMB_TIP,
    "index": INDEX_TIP,
    "middle": MIDDLE_TIP,
    "ring": RING_TIP,
    "pinky": PINKY_TIP,
}


FINGER_PIP_INDICES = {
    "index": INDEX_PIP,
    "middle": MIDDLE_PIP,
    "ring": RING_PIP,
    "pinky": PINKY_PIP,
}


FINGER_MCP_INDICES = {
    "index": INDEX_MCP,
    "middle": MIDDLE_MCP,
    "ring": RING_MCP,
    "pinky": PINKY_MCP,
}


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class HandLandmark:
    """
    One hand landmark.

    x/y are normalized values when the backend provides normalized
    coordinates.

    pixel_x/pixel_y are calculated image coordinates when image
    dimensions are available.
    """

    index: int

    x: float

    y: float

    z: float = 0.0

    visibility: float = 1.0

    pixel_x: Optional[int] = None

    pixel_y: Optional[int] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:

        return {
            "index": self.index,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "visibility": self.visibility,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "metadata": dict(self.metadata),
        }


@dataclass
class HandDetection:
    """
    One detected hand.
    """

    hand_id: int

    confidence: float

    handedness: str

    box: BoundingBox

    landmarks: list[
        HandLandmark
    ] = field(
        default_factory=list
    )

    gesture_hint: Optional[str] = None

    track_id: Optional[int] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def center(self) -> tuple[int, int]:

        return self.box.center

    @property
    def landmark_count(self) -> int:

        return len(self.landmarks)

    def get_landmark(
        self,
        index: int,
    ) -> Optional[HandLandmark]:

        for landmark in self.landmarks:

            if landmark.index == index:

                return landmark

        return None

    def get_fingertip(
        self,
        finger: str,
    ) -> Optional[HandLandmark]:

        index = FINGERTIP_INDICES.get(
            finger.lower().strip()
        )

        if index is None:
            return None

        return self.get_landmark(
            index
        )

    def get_fingertip_position(
        self,
        finger: str,
    ) -> Optional[
        tuple[int, int]
    ]:

        landmark = self.get_fingertip(
            finger
        )

        if landmark is None:
            return None

        if (
            landmark.pixel_x is not None
            and landmark.pixel_y is not None
        ):

            return (
                landmark.pixel_x,
                landmark.pixel_y,
            )

        return None

    def palm_center(
        self,
    ) -> tuple[float, float]:

        indices = [
            WRIST,
            INDEX_MCP,
            MIDDLE_MCP,
            RING_MCP,
            PINKY_MCP,
        ]

        points = []

        for index in indices:

            landmark = self.get_landmark(
                index
            )

            if landmark is not None:

                points.append(
                    (
                        landmark.x,
                        landmark.y,
                    )
                )

        if not points:

            return (
                float(self.center[0]),
                float(self.center[1]),
            )

        return (
            sum(
                point[0]
                for point in points
            )
            / len(points),
            sum(
                point[1]
                for point in points
            )
            / len(points),
        )

    def finger_states(
        self,
    ) -> dict[str, bool]:

        """
        Basic geometric finger-extension estimation.

        This is intentionally only a geometric hint.
        Full gesture interpretation belongs to gestures/.
        """

        states: dict[
            str,
            bool,
        ] = {}

        # Four non-thumb fingers.
        for finger in (
            "index",
            "middle",
            "ring",
            "pinky",
        ):

            tip = self.get_landmark(
                FINGERTIP_INDICES[finger]
            )

            pip = self.get_landmark(
                FINGER_PIP_INDICES[finger]
            )

            mcp = self.get_landmark(
                FINGER_MCP_INDICES[finger]
            )

            if (
                tip is None
                or pip is None
                or mcp is None
            ):

                states[finger] = False

                continue

            # Compare distance from wrist.
            wrist = self.get_landmark(
                WRIST
            )

            if wrist is None:

                states[finger] = (
                    tip.y < pip.y < mcp.y
                )

                continue

            tip_distance = _distance_2d(
                tip.x,
                tip.y,
                wrist.x,
                wrist.y,
            )

            pip_distance = _distance_2d(
                pip.x,
                pip.y,
                wrist.x,
                wrist.y,
            )

            states[finger] = (
                tip_distance
                > pip_distance * 1.08
            )

        # Thumb uses a different geometry.
        thumb_tip = self.get_landmark(
            THUMB_TIP
        )

        thumb_ip = self.get_landmark(
            THUMB_IP
        )

        thumb_mcp = self.get_landmark(
            THUMB_MCP
        )

        if (
            thumb_tip is not None
            and thumb_ip is not None
            and thumb_mcp is not None
        ):

            thumb_extension = (
                _distance_2d(
                    thumb_tip.x,
                    thumb_tip.y,
                    thumb_mcp.x,
                    thumb_mcp.y,
                )
                > _distance_2d(
                    thumb_ip.x,
                    thumb_ip.y,
                    thumb_mcp.x,
                    thumb_mcp.y,
                )
                * 1.05
            )

            states["thumb"] = (
                thumb_extension
            )

        else:

            states["thumb"] = False

        return states

    def to_dict(self) -> dict[str, Any]:

        return {
            "hand_id": self.hand_id,
            "confidence": self.confidence,
            "handedness": self.handedness,
            "box": self.box.to_dict(),
            "landmarks": [
                landmark.to_dict()
                for landmark in self.landmarks
            ],
            "gesture_hint": self.gesture_hint,
            "track_id": self.track_id,
            "landmark_count": self.landmark_count,
            "center": self.center,
            "metadata": dict(self.metadata),
        }


@dataclass
class HandDetectionResult:
    """
    Complete hand-detection result for one frame.
    """

    hands: list[
        HandDetection
    ]

    timestamp: float

    processing_time: float

    image_width: int

    image_height: int

    success: bool = True

    error: Optional[str] = None

    backend: str = "unavailable"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def count(self) -> int:

        return len(self.hands)

    @property
    def left_hands(self) -> list[
        HandDetection
    ]:

        return [
            hand
            for hand in self.hands
            if hand.handedness.lower()
            == "left"
        ]

    @property
    def right_hands(self) -> list[
        HandDetection
    ]:

        return [
            hand
            for hand in self.hands
            if hand.handedness.lower()
            == "right"
        ]

    @property
    def primary_hand(
        self,
    ) -> Optional[HandDetection]:

        if not self.hands:
            return None

        return max(
            self.hands,
            key=lambda hand:
            hand.confidence
            * max(1, hand.box.area),
        )

    def get_by_id(
        self,
        hand_id: int,
    ) -> Optional[HandDetection]:

        for hand in self.hands:

            if hand.hand_id == hand_id:

                return hand

        return None

    def to_dict(self) -> dict[str, Any]:

        return {
            "hands": [
                hand.to_dict()
                for hand in self.hands
            ],
            "count": self.count,
            "timestamp": self.timestamp,
            "processing_time": self.processing_time,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "success": self.success,
            "error": self.error,
            "backend": self.backend,
            "metadata": dict(self.metadata),
        }


@dataclass
class HandDetectionConfig:
    """
    Hand detector configuration.
    """

    confidence_threshold: float = 0.50

    max_hands: int = 2

    backend: str = "auto"

    model_path: Optional[str] = None

    static_image_mode: bool = False

    refine_landmarks: bool = True

    enable_handedness: bool = True

    enable_landmarks: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# BACKEND PROTOCOL
# ============================================================================


class HandDetectionBackend(Protocol):
    """
    Backend interface.

    A MediaPipe, YOLO, ONNX or custom hand detector can implement this
    interface.
    """

    @property
    def name(self) -> str:
        ...

    def is_available(self) -> bool:
        ...

    def detect(
        self,
        image: Any,
        confidence_threshold: float,
        max_hands: int,
    ) -> list[HandDetection]:
        ...


# ============================================================================
# GENERIC BACKEND
# ============================================================================


class GenericHandBackend:
    """
    Adapter for external hand detectors.

    The supplied detector may return:
    - HandDetection
    - list[HandDetection]
    - dictionaries
    """

    def __init__(
        self,
        detector: Any,
        name: str = "custom",
    ) -> None:

        self.detector = detector

        self._name = name

    @property
    def name(self) -> str:

        return self._name

    def is_available(self) -> bool:

        return self.detector is not None

    def detect(
        self,
        image: Any,
        confidence_threshold: float,
        max_hands: int,
    ) -> list[HandDetection]:

        if self.detector is None:

            return []

        raw = self.detector(
            image,
            confidence_threshold=(
                confidence_threshold
            ),
            max_hands=max_hands,
        )

        return self._normalize(
            raw
        )

    def _normalize(
        self,
        raw: Any,
    ) -> list[HandDetection]:

        if raw is None:

            return []

        if isinstance(
            raw,
            HandDetection,
        ):

            return [raw]

        if isinstance(
            raw,
            dict,
        ):

            raw = [raw]

        result: list[
            HandDetection
        ] = []

        for index, item in enumerate(
            raw
        ):

            if isinstance(
                item,
                HandDetection,
            ):

                result.append(item)

                continue

            if not isinstance(
                item,
                dict,
            ):

                continue

            try:

                result.append(
                    self._dict_to_hand(
                        item,
                        index,
                    )
                )

            except Exception:

                logger.debug(
                    "Unable to normalize "
                    "hand detection.",
                    exc_info=True,
                )

        return result

    @staticmethod
    def _dict_to_hand(
        item: dict[str, Any],
        default_id: int,
    ) -> HandDetection:

        hand_id = int(
            item.get(
                "hand_id",
                item.get(
                    "id",
                    default_id,
                ),
            )
        )

        confidence = float(
            item.get(
                "confidence",
                item.get(
                    "score",
                    0.0,
                ),
            )
        )

        handedness = str(
            item.get(
                "handedness",
                item.get(
                    "side",
                    "unknown",
                ),
            )
        )

        box = GenericHandBackend._parse_box(
            item.get(
                "box",
                item,
            )
        )

        if box is None:

            raise ValueError(
                "Hand detection has no valid box."
            )

        landmarks = (
            GenericHandBackend
            ._parse_landmarks(
                item.get(
                    "landmarks",
                    [],
                )
            )
        )

        return HandDetection(
            hand_id=hand_id,
            confidence=confidence,
            handedness=handedness,
            box=box,
            landmarks=landmarks,
            gesture_hint=item.get(
                "gesture_hint"
            ),
            track_id=item.get(
                "track_id"
            ),
            metadata=dict(
                item.get(
                    "metadata",
                    {},
                )
            ),
        )

    @staticmethod
    def _parse_box(
        value: Any,
    ) -> Optional[BoundingBox]:

        if isinstance(
            value,
            BoundingBox,
        ):

            return value

        if isinstance(
            value,
            dict,
        ):

            if {
                "x",
                "y",
                "width",
                "height",
            }.issubset(value):

                return BoundingBox(
                    x=value["x"],
                    y=value["y"],
                    width=value["width"],
                    height=value["height"],
                )

            if {
                "x1",
                "y1",
                "x2",
                "y2",
            }.issubset(value):

                return BoundingBox(
                    x=value["x1"],
                    y=value["y1"],
                    width=(
                        value["x2"]
                        - value["x1"]
                    ),
                    height=(
                        value["y2"]
                        - value["y1"]
                    ),
                )

        try:

            values = list(value)

            if len(values) == 4:

                x1, y1, x2, y2 = values

                return BoundingBox(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                )

        except Exception:
            pass

        return None

    @staticmethod
    def _parse_landmarks(
        value: Any,
    ) -> list[HandLandmark]:

        if value is None:

            return []

        result: list[
            HandLandmark
        ] = []

        for index, item in enumerate(
            value
        ):

            if isinstance(
                item,
                HandLandmark,
            ):

                result.append(item)

                continue

            if isinstance(
                item,
                dict,
            ):

                result.append(
                    HandLandmark(
                        index=int(
                            item.get(
                                "index",
                                index,
                            )
                        ),
                        x=float(
                            item.get(
                                "x",
                                0.0,
                            )
                        ),
                        y=float(
                            item.get(
                                "y",
                                0.0,
                            )
                        ),
                        z=float(
                            item.get(
                                "z",
                                0.0,
                            )
                        ),
                        visibility=float(
                            item.get(
                                "visibility",
                                1.0,
                            )
                        ),
                        pixel_x=(
                            item.get(
                                "pixel_x"
                            )
                        ),
                        pixel_y=(
                            item.get(
                                "pixel_y"
                            )
                        ),
                    )
                )

                continue

            try:

                values = list(item)

                if len(values) >= 2:

                    result.append(
                        HandLandmark(
                            index=index,
                            x=float(
                                values[0]
                            ),
                            y=float(
                                values[1]
                            ),
                            z=(
                                float(
                                    values[2]
                                )
                                if len(values)
                                >= 3
                                else 0.0
                            ),
                        )
                    )

            except Exception:

                continue

        return result


# ============================================================================
# HAND DETECTOR
# ============================================================================


class HandDetector:
    """
    Main RENIX hand-detection service.
    """

    def __init__(
        self,
        config: Optional[
            HandDetectionConfig
        ] = None,
        backend: Optional[
            HandDetectionBackend
        ] = None,
    ) -> None:

        self.config = (
            config
            or HandDetectionConfig()
        )

        self._backend = backend

        self._lock = threading.RLock()

        self._initialized = False

        self._last_result: Optional[
            HandDetectionResult
        ] = None

        self._frame_counter = 0

        self._initialize()

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def _initialize(self) -> None:

        self._initialized = True

        if self._backend is not None:

            logger.info(
                "RENIX hand detector initialized "
                "with backend: %s",
                self._backend.name,
            )

            return

        logger.info(
            "No hand-detection backend configured. "
            "Running in fallback mode."
        )

    # ========================================================================
    # STATUS
    # ========================================================================

    @property
    def backend_name(self) -> str:

        if self._backend is None:

            return "unavailable"

        return self._backend.name

    def is_available(self) -> bool:

        return (
            self._backend is not None
            and self._backend.is_available()
        )

    # ========================================================================
    # DETECTION
    # ========================================================================

    def detect(
        self,
        image: Any,
    ) -> HandDetectionResult:

        started = time.perf_counter()

        timestamp = time.time()

        with self._lock:

            self._frame_counter += 1

        width, height = (
            self._get_dimensions(
                image
            )
        )

        if image is None:

            return self._failure(
                timestamp,
                started,
                "Input image is None.",
                width,
                height,
            )

        if width <= 0 or height <= 0:

            return self._failure(
                timestamp,
                started,
                "Invalid image dimensions.",
                width,
                height,
            )

        if not self.is_available():

            result = HandDetectionResult(
                hands=[],
                timestamp=timestamp,
                processing_time=(
                    time.perf_counter()
                    - started
                ),
                image_width=width,
                image_height=height,
                success=True,
                backend="unavailable",
                metadata={
                    "fallback": True,
                    "reason": (
                        "No hand-detection "
                        "backend configured."
                    ),
                },
            )

            self._store_result(
                result
            )

            return result

        try:

            hands = self._backend.detect(
                image,
                self.config.confidence_threshold,
                self.config.max_hands,
            )

            hands = self._sanitize(
                hands
            )

            if self.config.enable_landmarks:

                self._convert_landmarks_to_pixels(
                    hands,
                    width,
                    height,
                )

            else:

                for hand in hands:

                    hand.landmarks.clear()

            if not self.config.enable_handedness:

                for hand in hands:

                    hand.handedness = (
                        "unknown"
                    )

            if (
                self.config.max_hands > 0
            ):

                hands = hands[
                    : self.config.max_hands
                ]

            result = HandDetectionResult(
                hands=hands,
                timestamp=timestamp,
                processing_time=(
                    time.perf_counter()
                    - started
                ),
                image_width=width,
                image_height=height,
                success=True,
                backend=self.backend_name,
                metadata={
                    "frame_number": (
                        self._frame_counter
                    ),
                },
            )

            self._store_result(
                result
            )

            return result

        except Exception as exc:

            logger.exception(
                "Hand detection failed."
            )

            return self._failure(
                timestamp,
                started,
                str(exc),
                width,
                height,
            )

    # ========================================================================
    # SANITIZATION
    # ========================================================================

    def _sanitize(
        self,
        hands: Sequence[
            HandDetection
        ],
    ) -> list[HandDetection]:

        result: list[
            HandDetection
        ] = []

        threshold = (
            self.config.confidence_threshold
        )

        for index, hand in enumerate(
            hands
        ):

            if not isinstance(
                hand,
                HandDetection,
            ):

                continue

            if hand.confidence < threshold:

                continue

            if (
                hand.box.width <= 0
                or hand.box.height <= 0
            ):

                continue

            hand.hand_id = int(
                hand.hand_id
            )

            hand.confidence = max(
                0.0,
                min(
                    1.0,
                    float(
                        hand.confidence
                    ),
                ),
            )

            hand.handedness = (
                str(
                    hand.handedness
                ).strip().lower()
                or "unknown"
            )

            if hand.handedness not in {
                "left",
                "right",
                "unknown",
            }:

                hand.handedness = (
                    "unknown"
                )

            result.append(
                hand
            )

        return result

    # ========================================================================
    # LANDMARK CONVERSION
    # ========================================================================

    @staticmethod
    def _convert_landmarks_to_pixels(
        hands: list[HandDetection],
        width: int,
        height: int,
    ) -> None:

        for hand in hands:

            for landmark in (
                hand.landmarks
            ):

                # Normalized coordinates.
                if (
                    0.0 <= landmark.x <= 1.0
                    and 0.0 <= landmark.y <= 1.0
                ):

                    landmark.pixel_x = int(
                        landmark.x
                        * width
                    )

                    landmark.pixel_y = int(
                        landmark.y
                        * height
                    )

                else:

                    # Backend may already return
                    # pixel coordinates.
                    landmark.pixel_x = int(
                        landmark.x
                    )

                    landmark.pixel_y = int(
                        landmark.y
                    )

    @staticmethod
    def _get_dimensions(
        image: Any,
    ) -> tuple[int, int]:

        if image is None:

            return 0, 0

        try:

            height, width = (
                image.shape[:2]
            )

            return (
                int(width),
                int(height),
            )

        except Exception:

            return 0, 0

    # ========================================================================
    # RESULT
    # ========================================================================

    def _store_result(
        self,
        result: HandDetectionResult,
    ) -> None:

        with self._lock:

            self._last_result = result

    def _failure(
        self,
        timestamp: float,
        started: float,
        error: str,
        width: int,
        height: int,
    ) -> HandDetectionResult:

        result = HandDetectionResult(
            hands=[],
            timestamp=timestamp,
            processing_time=(
                time.perf_counter()
                - started
            ),
            image_width=width,
            image_height=height,
            success=False,
            error=error,
            backend=self.backend_name,
        )

        self._store_result(
            result
        )

        return result

    def get_last_result(
        self,
    ) -> Optional[
        HandDetectionResult
    ]:

        with self._lock:

            return self._last_result

    # ========================================================================
    # BACKEND MANAGEMENT
    # ========================================================================

    def set_backend(
        self,
        backend: Optional[
            HandDetectionBackend
        ],
    ) -> None:

        with self._lock:

            self._backend = backend

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    def set_confidence_threshold(
        self,
        threshold: float,
    ) -> None:

        threshold = float(
            threshold
        )

        if not 0.0 <= threshold <= 1.0:

            raise ValueError(
                "Confidence threshold must "
                "be between 0 and 1."
            )

        with self._lock:

            self.config.confidence_threshold = (
                threshold
            )

    def set_max_hands(
        self,
        max_hands: int,
    ) -> None:

        max_hands = int(
            max_hands
        )

        if max_hands <= 0:

            raise ValueError(
                "max_hands must be positive."
            )

        with self._lock:

            self.config.max_hands = (
                max_hands
            )

    # ========================================================================
    # GEOMETRY HELPERS
    # ========================================================================

    @staticmethod
    def distance_between_landmarks(
        hand: HandDetection,
        first_index: int,
        second_index: int,
    ) -> Optional[float]:

        first = hand.get_landmark(
            first_index
        )

        second = hand.get_landmark(
            second_index
        )

        if (
            first is None
            or second is None
        ):

            return None

        return _distance_3d(
            first.x,
            first.y,
            first.z,
            second.x,
            second.y,
            second.z,
        )

    @staticmethod
    def distance_between_pixels(
        hand: HandDetection,
        first_index: int,
        second_index: int,
    ) -> Optional[float]:

        first = hand.get_landmark(
            first_index
        )

        second = hand.get_landmark(
            second_index
        )

        if (
            first is None
            or second is None
            or first.pixel_x is None
            or first.pixel_y is None
            or second.pixel_x is None
            or second.pixel_y is None
        ):

            return None

        return _distance_2d(
            first.pixel_x,
            first.pixel_y,
            second.pixel_x,
            second.pixel_y,
        )

    @staticmethod
    def palm_size(
        hand: HandDetection,
    ) -> float:

        wrist = hand.get_landmark(
            WRIST
        )

        middle_mcp = hand.get_landmark(
            MIDDLE_MCP
        )

        if (
            wrist is None
            or middle_mcp is None
        ):

            return float(
                max(
                    hand.box.width,
                    hand.box.height,
                )
            )

        return _distance_2d(
            wrist.x,
            wrist.y,
            middle_mcp.x,
            middle_mcp.y,
        )

    @staticmethod
    def is_hand_open(
        hand: HandDetection,
    ) -> bool:

        states = hand.finger_states()

        return all(
            states.get(
                finger,
                False,
            )
            for finger in (
                "thumb",
                "index",
                "middle",
                "ring",
                "pinky",
            )
        )

    @staticmethod
    def is_fist(
        hand: HandDetection,
    ) -> bool:

        states = hand.finger_states()

        return not any(
            states.get(
                finger,
                False,
            )
            for finger in (
                "index",
                "middle",
                "ring",
                "pinky",
            )
        )

    @staticmethod
    def is_pointing(
        hand: HandDetection,
    ) -> bool:

        states = hand.finger_states()

        return (
            states.get("index", False)
            and not states.get(
                "middle",
                False,
            )
            and not states.get(
                "ring",
                False,
            )
            and not states.get(
                "pinky",
                False,
            )
        )

    @staticmethod
    def is_pinch_candidate(
        hand: HandDetection,
        threshold: float = 0.08,
    ) -> bool:

        thumb = hand.get_landmark(
            THUMB_TIP
        )

        index = hand.get_landmark(
            INDEX_TIP
        )

        if (
            thumb is None
            or index is None
        ):

            return False

        distance = _distance_2d(
            thumb.x,
            thumb.y,
            index.x,
            index.y,
        )

        return distance <= threshold


# ============================================================================
# SHARED DETECTOR
# ============================================================================


_default_detector: Optional[
    HandDetector
] = None

_default_detector_lock = (
    threading.RLock()
)


def get_hand_detector() -> HandDetector:

    global _default_detector

    with _default_detector_lock:

        if _default_detector is None:

            _default_detector = (
                HandDetector()
            )

        return _default_detector


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def detect_hands(
    image: Any,
) -> HandDetectionResult:

    return get_hand_detector().detect(
        image
    )


def get_hands(
    image: Any,
) -> list[HandDetection]:

    return detect_hands(
        image
    ).hands


def get_primary_hand(
    image: Any,
) -> Optional[HandDetection]:

    return detect_hands(
        image
    ).primary_hand


def is_hand_open(
    hand: HandDetection,
) -> bool:

    return HandDetector.is_hand_open(
        hand
    )


def is_fist(
    hand: HandDetection,
) -> bool:

    return HandDetector.is_fist(
        hand
    )


def is_pointing(
    hand: HandDetection,
) -> bool:

    return HandDetector.is_pointing(
        hand
    )


def is_pinch_candidate(
    hand: HandDetection,
    threshold: float = 0.08,
) -> bool:

    return HandDetector.is_pinch_candidate(
        hand,
        threshold,
    )


# ============================================================================
# GEOMETRY FUNCTIONS
# ============================================================================


def _distance_2d(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:

    return math.sqrt(
        (x2 - x1) ** 2
        + (y2 - y1) ** 2
    )


def _distance_3d(
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
) -> float:

    return math.sqrt(
        (x2 - x1) ** 2
        + (y2 - y1) ** 2
        + (z2 - z1) ** 2
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_MCP",
    "INDEX_PIP",
    "INDEX_DIP",
    "INDEX_TIP",
    "MIDDLE_MCP",
    "MIDDLE_PIP",
    "MIDDLE_DIP",
    "MIDDLE_TIP",
    "RING_MCP",
    "RING_PIP",
    "RING_DIP",
    "RING_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
    "HandLandmark",
    "HandDetection",
    "HandDetectionResult",
    "HandDetectionConfig",
    "HandDetectionBackend",
    "GenericHandBackend",
    "HandDetector",
    "get_hand_detector",
    "detect_hands",
    "get_hands",
    "get_primary_hand",
    "is_hand_open",
    "is_fist",
    "is_pointing",
    "is_pinch_candidate",
]


