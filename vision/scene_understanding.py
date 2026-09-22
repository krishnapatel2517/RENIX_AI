"""
RENIX Vision — Scene Understanding

Combines outputs from the vision subsystem to create a structured
understanding of the current visual scene.

Responsibilities:
- Combine faces, people, objects, poses, OCR and screen information
- Build a structured scene representation
- Identify the primary subject
- Estimate scene type
- Describe spatial relationships
- Track scene changes
- Provide context to RENIX core/orchestrator
- Remain backend-agnostic

This module does NOT:
- perform raw camera capture
- perform low-level face detection
- perform raw object detection
- execute computer actions
- control gestures

Those responsibilities belong to their respective modules.
"""

from __future__ import annotations

import logging
import math
import threading
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

from .object_detection import (
    BoundingBox,
    ObjectDetection,
)
from .pose_detection import (
    PoseDetection,
    PoseLandmark,
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
)


logger = logging.getLogger(
    "RENIX.vision.scene_understanding"
)


# ============================================================================
# ENUMS
# ============================================================================


class SceneType(str, Enum):
    UNKNOWN = "unknown"
    EMPTY = "empty"
    PERSON = "person"
    PEOPLE = "people"
    DESKTOP = "desktop"
    DOCUMENT = "document"
    ROOM = "room"
    OUTDOOR = "outdoor"
    VEHICLE = "vehicle"
    SPORTS = "sports"
    CLASSROOM = "classroom"
    OFFICE = "office"
    HOME = "home"
    STREET = "street"
    SCREEN = "screen"
    PRODUCT = "product"
    ANIMAL = "animal"
    MIXED = "mixed"


class SubjectType(str, Enum):
    UNKNOWN = "unknown"
    PERSON = "person"
    OBJECT = "object"
    FACE = "face"
    DOCUMENT = "document"
    SCREEN = "screen"
    ANIMAL = "animal"


class SpatialRelation(str, Enum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    NEAR = "near"
    FAR = "far"
    INSIDE = "inside"
    CONTAINS = "contains"
    OVERLAPPING = "overlapping"
    INTERSECTING = "intersecting"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class SceneSubject:
    """
    A normalized entity inside a scene.
    """

    subject_id: str

    subject_type: SubjectType

    label: str

    confidence: float = 1.0

    box: Optional[BoundingBox] = None

    center: Optional[tuple[float, float]] = None

    source: str = "unknown"

    track_id: Optional[int] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "subject_type": self.subject_type.value,
            "label": self.label,
            "confidence": self.confidence,
            "box": (
                self.box.to_dict()
                if self.box is not None
                else None
            ),
            "center": self.center,
            "source": self.source,
            "track_id": self.track_id,
            "metadata": dict(self.metadata),
        }


@dataclass
class SceneRelation:
    """
    Spatial or semantic relationship between two scene subjects.
    """

    subject_a: str

    relation: SpatialRelation

    subject_b: str

    confidence: float = 1.0

    distance: Optional[float] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_a": self.subject_a,
            "relation": self.relation.value,
            "subject_b": self.subject_b,
            "confidence": self.confidence,
            "distance": self.distance,
            "metadata": dict(self.metadata),
        }


@dataclass
class SceneText:
    """
    OCR/text found in the scene.
    """

    text: str

    confidence: float = 1.0

    box: Optional[BoundingBox] = None

    source: str = "ocr"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "box": (
                self.box.to_dict()
                if self.box is not None
                else None
            ),
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass
class SceneDescription:
    """
    High-level scene interpretation.
    """

    scene_type: SceneType = SceneType.UNKNOWN

    description: str = ""

    confidence: float = 0.0

    person_count: int = 0

    object_count: int = 0

    face_count: int = 0

    text_count: int = 0

    has_screen: bool = False

    has_document: bool = False

    has_human_pose: bool = False

    dominant_objects: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_type": self.scene_type.value,
            "description": self.description,
            "confidence": self.confidence,
            "person_count": self.person_count,
            "object_count": self.object_count,
            "face_count": self.face_count,
            "text_count": self.text_count,
            "has_screen": self.has_screen,
            "has_document": self.has_document,
            "has_human_pose": self.has_human_pose,
            "dominant_objects": list(
                self.dominant_objects
            ),
            "metadata": dict(self.metadata),
        }


@dataclass
class SceneUnderstandingResult:
    """
    Complete visual understanding of one frame.
    """

    timestamp: float

    processing_time: float

    image_width: int

    image_height: int

    subjects: list[SceneSubject] = field(
        default_factory=list
    )

    relations: list[SceneRelation] = field(
        default_factory=list
    )

    texts: list[SceneText] = field(
        default_factory=list
    )

    description: SceneDescription = field(
        default_factory=SceneDescription
    )

    success: bool = True

    error: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def primary_subject(
        self,
    ) -> Optional[SceneSubject]:

        if not self.subjects:
            return None

        return max(
            self.subjects,
            key=lambda subject: subject.confidence,
        )

    @property
    def people(
        self,
    ) -> list[SceneSubject]:

        return [
            subject
            for subject in self.subjects
            if subject.subject_type
            == SubjectType.PERSON
        ]

    @property
    def objects(
        self,
    ) -> list[SceneSubject]:

        return [
            subject
            for subject in self.subjects
            if subject.subject_type
            == SubjectType.OBJECT
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "processing_time": self.processing_time,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "subjects": [
                subject.to_dict()
                for subject in self.subjects
            ],
            "relations": [
                relation.to_dict()
                for relation in self.relations
            ],
            "texts": [
                text.to_dict()
                for text in self.texts
            ],
            "description": (
                self.description.to_dict()
            ),
            "success": self.success,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# SCENE UNDERSTANDING CONFIG
# ============================================================================


@dataclass
class SceneUnderstandingConfig:
    """
    Configuration for scene interpretation.
    """

    relation_distance_threshold: float = 0.15

    overlap_threshold: float = 0.20

    max_subjects: int = 100

    max_text_items: int = 50

    generate_description: bool = True

    infer_scene_type: bool = True

    calculate_relations: bool = True

    include_low_confidence: bool = False

    confidence_threshold: float = 0.35

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# SCENE UNDERSTANDING ENGINE
# ============================================================================


class SceneUnderstandingEngine:
    """
    Main RENIX scene-understanding service.
    """

    def __init__(
        self,
        config: Optional[
            SceneUnderstandingConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or SceneUnderstandingConfig()
        )

        self._lock = threading.RLock()

        self._last_result: Optional[
            SceneUnderstandingResult
        ] = None

        self._frame_counter = 0

        logger.info(
            "RENIX scene understanding initialized."
        )

    # ========================================================================
    # MAIN PROCESSING
    # ========================================================================

    def understand(
        self,
        image: Any = None,
        *,
        object_detections: Optional[
            Sequence[ObjectDetection]
        ] = None,
        pose_detections: Optional[
            Sequence[PoseDetection]
        ] = None,
        face_detections: Optional[
            Sequence[Any]
        ] = None,
        person_detections: Optional[
            Sequence[Any]
        ] = None,
        ocr_results: Optional[
            Sequence[Any]
        ] = None,
        screen_information: Optional[
            dict[str, Any]
        ] = None,
        document_information: Optional[
            dict[str, Any]
        ] = None,
    ) -> SceneUnderstandingResult:

        started = time.perf_counter()

        timestamp = time.time()

        width, height = (
            self._get_dimensions(image)
        )

        with self._lock:
            self._frame_counter += 1

        try:

            subjects: list[
                SceneSubject
            ] = []

            texts: list[
                SceneText
            ] = []

            subjects.extend(
                self._normalize_objects(
                    object_detections
                )
            )

            subjects.extend(
                self._normalize_people(
                    person_detections
                )
            )

            subjects.extend(
                self._normalize_faces(
                    face_detections
                )
            )

            subjects.extend(
                self._normalize_poses(
                    pose_detections
                )
            )

            texts.extend(
                self._normalize_text(
                    ocr_results
                )
            )

            subjects = (
                self._deduplicate_subjects(
                    subjects
                )
            )

            if (
                self.config.max_subjects
                > 0
            ):
                subjects = subjects[
                    : self.config.max_subjects
                ]

            if (
                self.config.max_text_items
                > 0
            ):
                texts = texts[
                    : self.config.max_text_items
                ]

            relations: list[
                SceneRelation
            ] = []

            if self.config.calculate_relations:

                relations = (
                    self._calculate_relations(
                        subjects
                    )
                )

            description = (
                self._build_description(
                    subjects=subjects,
                    texts=texts,
                    pose_detections=(
                        pose_detections
                        or []
                    ),
                    face_detections=(
                        face_detections
                        or []
                    ),
                    screen_information=(
                        screen_information
                    ),
                    document_information=(
                        document_information
                    ),
                )
            )

            result = SceneUnderstandingResult(
                timestamp=timestamp,
                processing_time=(
                    time.perf_counter()
                    - started
                ),
                image_width=width,
                image_height=height,
                subjects=subjects,
                relations=relations,
                texts=texts,
                description=description,
                success=True,
                metadata={
                    "frame_number": (
                        self._frame_counter
                    ),
                    "screen_information": (
                        screen_information
                        or {}
                    ),
                    "document_information": (
                        document_information
                        or {}
                    ),
                },
            )

            self._store_result(result)

            return result

        except Exception as exc:

            logger.exception(
                "Scene understanding failed."
            )

            result = (
                SceneUnderstandingResult(
                    timestamp=timestamp,
                    processing_time=(
                        time.perf_counter()
                        - started
                    ),
                    image_width=width,
                    image_height=height,
                    success=False,
                    error=str(exc),
                )
            )

            self._store_result(result)

            return result

    # ========================================================================
    # OBJECT NORMALIZATION
    # ========================================================================

    def _normalize_objects(
        self,
        detections: Optional[
            Sequence[ObjectDetection]
        ],
    ) -> list[SceneSubject]:

        if not detections:
            return []

        result: list[
            SceneSubject
        ] = []

        for index, detection in enumerate(
            detections
        ):

            if not isinstance(
                detection,
                ObjectDetection,
            ):
                continue

            confidence = float(
                detection.confidence
            )

            if (
                not self.config.include_low_confidence
                and confidence
                < self.config.confidence_threshold
            ):
                continue

            center = self._box_center(
                detection.box
            )

            result.append(
                SceneSubject(
                    subject_id=(
                        f"object_{index}"
                    ),
                    subject_type=(
                        SubjectType.OBJECT
                    ),
                    label=detection.label,
                    confidence=confidence,
                    box=detection.box,
                    center=center,
                    source="object_detection",
                    track_id=getattr(
                        detection,
                        "track_id",
                        None,
                    ),
                    metadata={
                        "class_id": getattr(
                            detection,
                            "class_id",
                            None,
                        )
                    },
                )
            )

        return result

    # ========================================================================
    # PERSON NORMALIZATION
    # ========================================================================

    def _normalize_people(
        self,
        detections: Optional[
            Sequence[Any]
        ],
    ) -> list[SceneSubject]:

        if not detections:
            return []

        result: list[
            SceneSubject
        ] = []

        for index, detection in enumerate(
            detections
        ):

            if isinstance(
                detection,
                dict,
            ):

                confidence = float(
                    detection.get(
                        "confidence",
                        detection.get(
                            "score",
                            1.0,
                        ),
                    )
                )

                box = self._parse_box(
                    detection.get(
                        "box"
                    )
                )

                track_id = detection.get(
                    "track_id"
                )

            else:

                confidence = float(
                    getattr(
                        detection,
                        "confidence",
                        1.0,
                    )
                )

                box = self._parse_box(
                    getattr(
                        detection,
                        "box",
                        None,
                    )
                )

                track_id = getattr(
                    detection,
                    "track_id",
                    None,
                )

            if (
                not self.config.include_low_confidence
                and confidence
                < self.config.confidence_threshold
            ):
                continue

            result.append(
                SceneSubject(
                    subject_id=(
                        f"person_{index}"
                    ),
                    subject_type=(
                        SubjectType.PERSON
                    ),
                    label="person",
                    confidence=confidence,
                    box=box,
                    center=self._box_center(
                        box
                    ),
                    source="person_detection",
                    track_id=track_id,
                )
            )

        return result

    # ========================================================================
    # FACE NORMALIZATION
    # ========================================================================

    def _normalize_faces(
        self,
        detections: Optional[
            Sequence[Any]
        ],
    ) -> list[SceneSubject]:

        if not detections:
            return []

        result: list[
            SceneSubject
        ] = []

        for index, detection in enumerate(
            detections
        ):

            if isinstance(
                detection,
                dict,
            ):

                confidence = float(
                    detection.get(
                        "confidence",
                        detection.get(
                            "score",
                            1.0,
                        ),
                    )
                )

                label = str(
                    detection.get(
                        "name",
                        detection.get(
                            "label",
                            "face",
                        ),
                    )
                )

                box = self._parse_box(
                    detection.get(
                        "box"
                    )
                )

            else:

                confidence = float(
                    getattr(
                        detection,
                        "confidence",
                        1.0,
                    )
                )

                label = str(
                    getattr(
                        detection,
                        "name",
                        "face",
                    )
                )

                box = self._parse_box(
                    getattr(
                        detection,
                        "box",
                        None,
                    )
                )

            if (
                not self.config.include_low_confidence
                and confidence
                < self.config.confidence_threshold
            ):
                continue

            result.append(
                SceneSubject(
                    subject_id=(
                        f"face_{index}"
                    ),
                    subject_type=(
                        SubjectType.FACE
                    ),
                    label=label,
                    confidence=confidence,
                    box=box,
                    center=self._box_center(
                        box
                    ),
                    source="face_detection",
                )
            )

        return result

    # ========================================================================
    # POSE NORMALIZATION
    # ========================================================================

    def _normalize_poses(
        self,
        poses: Optional[
            Sequence[PoseDetection]
        ],
    ) -> list[SceneSubject]:

        if not poses:
            return []

        result: list[
            SceneSubject
        ] = []

        for index, pose in enumerate(
            poses
        ):

            if not isinstance(
                pose,
                PoseDetection,
            ):
                continue

            if (
                not self.config.include_low_confidence
                and pose.confidence
                < self.config.confidence_threshold
            ):
                continue

            box = self._pose_box(
                pose
            )

            center = (
                self._pose_center(
                    pose
                )
            )

            result.append(
                SceneSubject(
                    subject_id=(
                        f"pose_{index}"
                    ),
                    subject_type=(
                        SubjectType.PERSON
                    ),
                    label="person",
                    confidence=pose.confidence,
                    box=box,
                    center=center,
                    source="pose_detection",
                    track_id=pose.track_id,
                    metadata={
                        "landmark_count": (
                            pose.landmark_count
                        ),
                        "pose_id": pose.pose_id,
                    },
                )
            )

        return result

    # ========================================================================
    # OCR NORMALIZATION
    # ========================================================================

    def _normalize_text(
        self,
        results: Optional[
            Sequence[Any]
        ],
    ) -> list[SceneText]:

        if not results:
            return []

        output: list[
            SceneText
        ] = []

        for item in results:

            if isinstance(
                item,
                SceneText,
            ):

                output.append(item)
                continue

            if isinstance(
                item,
                str,
            ):

                if item.strip():

                    output.append(
                        SceneText(
                            text=item.strip()
                        )
                    )

                continue

            if isinstance(
                item,
                dict,
            ):

                text = str(
                    item.get(
                        "text",
                        "",
                    )
                ).strip()

                if not text:
                    continue

                confidence = float(
                    item.get(
                        "confidence",
                        item.get(
                            "score",
                            1.0,
                        ),
                    )
                )

                box = self._parse_box(
                    item.get(
                        "box"
                    )
                )

                output.append(
                    SceneText(
                        text=text,
                        confidence=confidence,
                        box=box,
                        source=str(
                            item.get(
                                "source",
                                "ocr",
                            )
                        ),
                    )
                )

        return output

    # ========================================================================
    # DESCRIPTION
    # ========================================================================

    def _build_description(
        self,
        *,
        subjects: Sequence[
            SceneSubject
        ],
        texts: Sequence[
            SceneText
        ],
        pose_detections: Sequence[
            PoseDetection
        ],
        face_detections: Sequence[Any],
        screen_information: Optional[
            dict[str, Any]
        ],
        document_information: Optional[
            dict[str, Any]
        ],
    ) -> SceneDescription:

        people = [
            subject
            for subject in subjects
            if subject.subject_type
            == SubjectType.PERSON
        ]

        objects = [
            subject
            for subject in subjects
            if subject.subject_type
            == SubjectType.OBJECT
        ]

        faces = [
            subject
            for subject in subjects
            if subject.subject_type
            == SubjectType.FACE
        ]

        has_screen = bool(
            screen_information
            and screen_information.get(
                "detected",
                True,
            )
        )

        has_document = bool(
            document_information
            and document_information.get(
                "detected",
                True,
            )
        )

        dominant_objects = self._dominant_objects(
            objects
        )

        scene_type = (
            self._infer_scene_type(
                people=people,
                objects=objects,
                texts=texts,
                has_screen=has_screen,
                has_document=has_document,
                pose_detections=(
                    pose_detections
                ),
            )
        )

        description = (
            self._generate_description(
                scene_type=scene_type,
                people=people,
                objects=objects,
                texts=texts,
                has_screen=has_screen,
                has_document=has_document,
                dominant_objects=(
                    dominant_objects
                ),
            )
        )

        confidence = self._scene_confidence(
            people=people,
            objects=objects,
            texts=texts,
            pose_detections=(
                pose_detections
            ),
            scene_type=scene_type,
        )

        return SceneDescription(
            scene_type=scene_type,
            description=description,
            confidence=confidence,
            person_count=len(people),
            object_count=len(objects),
            face_count=len(faces),
            text_count=len(texts),
            has_screen=has_screen,
            has_document=has_document,
            has_human_pose=bool(
                pose_detections
            ),
            dominant_objects=dominant_objects,
        )

    # ========================================================================
    # SCENE TYPE
    # ========================================================================

    def _infer_scene_type(
        self,
        *,
        people: Sequence[
            SceneSubject
        ],
        objects: Sequence[
            SceneSubject
        ],
        texts: Sequence[
            SceneText
        ],
        has_screen: bool,
        has_document: bool,
        pose_detections: Sequence[
            PoseDetection
        ],
    ) -> SceneType:

        if not self.config.infer_scene_type:
            return SceneType.UNKNOWN

        if (
            not people
            and not objects
            and not texts
            and not has_screen
            and not has_document
        ):
            return SceneType.EMPTY

        labels = {
            subject.label.lower()
            for subject in objects
        }

        if has_screen:
            return SceneType.SCREEN

        if has_document:
            return SceneType.DOCUMENT

        sports_terms = {
            "ball",
            "bat",
            "racket",
            "sports ball",
            "football",
            "baseball",
            "tennis racket",
            "cricket bat",
            "goal",
            "wicket",
        }

        if labels.intersection(
            sports_terms
        ):
            return SceneType.SPORTS

        classroom_terms = {
            "book",
            "laptop",
            "backpack",
            "school",
            "desk",
            "whiteboard",
            "blackboard",
            "chair",
        }

        if (
            len(people) >= 2
            and labels.intersection(
                classroom_terms
            )
        ):
            return SceneType.CLASSROOM

        office_terms = {
            "laptop",
            "keyboard",
            "mouse",
            "monitor",
            "desk",
            "printer",
            "computer",
            "office chair",
        }

        if labels.intersection(
            office_terms
        ):
            return SceneType.OFFICE

        home_terms = {
            "sofa",
            "couch",
            "bed",
            "television",
            "tv",
            "refrigerator",
            "microwave",
            "dining table",
            "kitchen",
        }

        if labels.intersection(
            home_terms
        ):
            return SceneType.HOME

        vehicle_terms = {
            "car",
            "bus",
            "truck",
            "motorcycle",
            "bicycle",
            "train",
            "airplane",
        }

        if labels.intersection(
            vehicle_terms
        ):
            return SceneType.VEHICLE

        animal_terms = {
            "dog",
            "cat",
            "horse",
            "bird",
            "cow",
            "sheep",
            "elephant",
            "bear",
        }

        if labels.intersection(
            animal_terms
        ):
            return SceneType.ANIMAL

        if len(people) == 1:
            return SceneType.PERSON

        if len(people) > 1:
            return SceneType.PEOPLE

        if objects:
            return SceneType.PRODUCT

        if texts:
            return SceneType.DOCUMENT

        return SceneType.MIXED

    # ========================================================================
    # DESCRIPTION GENERATION
    # ========================================================================

    def _generate_description(
        self,
        *,
        scene_type: SceneType,
        people: Sequence[
            SceneSubject
        ],
        objects: Sequence[
            SceneSubject
        ],
        texts: Sequence[
            SceneText
        ],
        has_screen: bool,
        has_document: bool,
        dominant_objects: Sequence[str],
    ) -> str:

        if scene_type == SceneType.EMPTY:
            return "No significant subjects detected."

        parts: list[str] = []

        if people:

            if len(people) == 1:
                parts.append(
                    "One person is visible."
                )

            else:
                parts.append(
                    f"{len(people)} people are visible."
                )

        if dominant_objects:

            object_text = ", ".join(
                dominant_objects[:5]
            )

            parts.append(
                f"Visible objects include "
                f"{object_text}."
            )

        if has_screen:
            parts.append(
                "A screen or display is present."
            )

        if has_document:
            parts.append(
                "A document or document-like "
                "surface is present."
            )

        if texts:

            sample = texts[0].text.strip()

            if len(sample) > 100:
                sample = sample[:97] + "..."

            if sample:
                parts.append(
                    f"Visible text includes "
                    f"\"{sample}\"."
                )

        if not parts:
            return (
                "The scene contains "
                "detected visual information."
            )

        return " ".join(parts)

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    @staticmethod
    def _scene_confidence(
        *,
        people: Sequence[
            SceneSubject
        ],
        objects: Sequence[
            SceneSubject
        ],
        texts: Sequence[
            SceneText
        ],
        pose_detections: Sequence[
            PoseDetection
        ],
        scene_type: SceneType,
    ) -> float:

        scores: list[float] = []

        scores.extend(
            subject.confidence
            for subject in people
        )

        scores.extend(
            subject.confidence
            for subject in objects
        )

        scores.extend(
            text.confidence
            for text in texts
        )

        scores.extend(
            pose.confidence
            for pose in pose_detections
        )

        if not scores:
            return (
                0.0
                if scene_type == SceneType.UNKNOWN
                else 0.25
            )

        return max(
            0.0,
            min(
                1.0,
                sum(scores)
                / len(scores),
            ),
        )

    # ========================================================================
    # RELATION CALCULATION
    # ========================================================================

    def _calculate_relations(
        self,
        subjects: Sequence[
            SceneSubject
        ],
    ) -> list[SceneRelation]:

        relations: list[
            SceneRelation
        ] = []

        for index, first in enumerate(
            subjects
        ):

            if first.center is None:
                continue

            for second in subjects[
                index + 1 :
            ]:

                if second.center is None:
                    continue

                relation_set = (
                    self._relations_between(
                        first,
                        second,
                    )
                )

                relations.extend(
                    relation_set
                )

        return relations

    def _relations_between(
        self,
        first: SceneSubject,
        second: SceneSubject,
    ) -> list[SceneRelation]:

        if (
            first.center is None
            or second.center is None
        ):
            return []

        x1, y1 = first.center
        x2, y2 = second.center

        dx = x2 - x1
        dy = y2 - y1

        distance = math.sqrt(
            dx * dx
            + dy * dy
        )

        relations: list[
            SceneRelation
        ] = []

        threshold = (
            self.config.relation_distance_threshold
        )

        if dx < -threshold:

            relations.append(
                SceneRelation(
                    subject_a=first.subject_id,
                    relation=(
                        SpatialRelation.RIGHT_OF
                    ),
                    subject_b=second.subject_id,
                    distance=distance,
                )
            )

            relations.append(
                SceneRelation(
                    subject_a=second.subject_id,
                    relation=(
                        SpatialRelation.LEFT_OF
                    ),
                    subject_b=first.subject_id,
                    distance=distance,
                )
            )

        elif dx > threshold:

            relations.append(
                SceneRelation(
                    subject_a=first.subject_id,
                    relation=(
                        SpatialRelation.LEFT_OF
                    ),
                    subject_b=second.subject_id,
                    distance=distance,
                )
            )

            relations.append(
                SceneRelation(
                    subject_a=second.subject_id,
                    relation=(
                        SpatialRelation.RIGHT_OF
                    ),
                    subject_b=first.subject_id,
                    distance=distance,
                )
            )

        if dy < -threshold:

            relations.append(
                SceneRelation(
                    subject_a=first.subject_id,
                    relation=(
                        SpatialRelation.BELOW
                    ),
                    subject_b=second.subject_id,
                    distance=distance,
                )
            )

            relations.append(
                SceneRelation(
                    subject_a=second.subject_id,
                    relation=(
                        SpatialRelation.ABOVE
                    ),
                    subject_b=first.subject_id,
                    distance=distance,
                )
            )

        elif dy > threshold:

            relations.append(
                SceneRelation(
                    subject_a=first.subject_id,
                    relation=(
                        SpatialRelation.ABOVE
                    ),
                    subject_b=second.subject_id,
                    distance=distance,
                )
            )

            relations.append(
                SceneRelation(
                    subject_a=second.subject_id,
                    relation=(
                        SpatialRelation.BELOW
                    ),
                    subject_b=first.subject_id,
                    distance=distance,
                )
            )

        if distance <= threshold:

            relations.append(
                SceneRelation(
                    subject_a=first.subject_id,
                    relation=(
                        SpatialRelation.NEAR
                    ),
                    subject_b=second.subject_id,
                    distance=distance,
                )
            )

        if self._boxes_overlap(
            first.box,
            second.box,
        ):

            relations.append(
                SceneRelation(
                    subject_a=first.subject_id,
                    relation=(
                        SpatialRelation.OVERLAPPING
                    ),
                    subject_b=second.subject_id,
                    distance=distance,
                )
            )

        return relations

    # ========================================================================
    # SUBJECT HELPERS
    # ========================================================================

    @staticmethod
    def _dominant_objects(
        objects: Sequence[
            SceneSubject
        ],
    ) -> list[str]:

        scores: dict[
            str,
            float,
        ] = {}

        for subject in objects:

            label = subject.label.strip()

            if not label:
                continue

            scores[label] = max(
                scores.get(
                    label,
                    0.0,
                ),
                subject.confidence,
            )

        return [
            label
            for label, _ in sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

    @staticmethod
    def _box_center(
        box: Optional[
            BoundingBox
        ],
    ) -> Optional[
        tuple[float, float]
    ]:

        if box is None:
            return None

        center = box.center

        return (
            float(center[0]),
            float(center[1]),
        )

    @staticmethod
    def _pose_center(
        pose: PoseDetection,
    ) -> Optional[
        tuple[float, float]
    ]:

        points: list[
            tuple[float, float]
        ] = []

        for index in (
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_HIP,
            RIGHT_HIP,
        ):

            landmark = pose.get_landmark(
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

            nose = pose.get_landmark(
                NOSE
            )

            if nose is not None:

                return (
                    nose.x,
                    nose.y,
                )

            return None

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

    @staticmethod
    def _pose_box(
        pose: PoseDetection,
    ) -> Optional[
        BoundingBox
    ]:

        if pose.box is not None:
            return pose.box

        valid = [
            landmark
            for landmark in pose.landmarks
            if landmark.x is not None
            and landmark.y is not None
        ]

        if not valid:
            return None

        min_x = min(
            landmark.x
            for landmark in valid
        )

        max_x = max(
            landmark.x
            for landmark in valid
        )

        min_y = min(
            landmark.y
            for landmark in valid
        )

        max_y = max(
            landmark.y
            for landmark in valid
        )

        return BoundingBox(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
        )

    # ========================================================================
    # DEDUPLICATION
    # ========================================================================

    def _deduplicate_subjects(
        self,
        subjects: Sequence[
            SceneSubject
        ],
    ) -> list[SceneSubject]:

        result: list[
            SceneSubject
        ] = []

        for subject in subjects:

            duplicate = False

            for existing in result:

                if (
                    subject.subject_type
                    != existing.subject_type
                ):
                    continue

                if (
                    subject.center is None
                    or existing.center is None
                ):
                    continue

                distance = math.sqrt(
                    (
                        subject.center[0]
                        - existing.center[0]
                    )
                    ** 2
                    + (
                        subject.center[1]
                        - existing.center[1]
                    )
                    ** 2
                )

                if distance < 0.03:

                    if (
                        subject.confidence
                        > existing.confidence
                    ):

                        result.remove(
                            existing
                        )
                        result.append(
                            subject
                        )

                    duplicate = True
                    break

            if not duplicate:
                result.append(subject)

        return result

    # ========================================================================
    # BOX OPERATIONS
    # ========================================================================

    @staticmethod
    def _boxes_overlap(
        first: Optional[
            BoundingBox
        ],
        second: Optional[
            BoundingBox
        ],
    ) -> bool:

        if first is None or second is None:
            return False

        first_x1 = first.x
        first_y1 = first.y
        first_x2 = (
            first.x + first.width
        )
        first_y2 = (
            first.y + first.height
        )

        second_x1 = second.x
        second_y1 = second.y
        second_x2 = (
            second.x + second.width
        )
        second_y2 = (
            second.y + second.height
        )

        intersection_width = max(
            0.0,
            min(
                first_x2,
                second_x2,
            )
            - max(
                first_x1,
                second_x1,
            ),
        )

        intersection_height = max(
            0.0,
            min(
                first_y2,
                second_y2,
            )
            - max(
                first_y1,
                second_y1,
            ),
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        if intersection_area <= 0:
            return False

        first_area = max(
            1e-9,
            first.width
            * first.height,
        )

        second_area = max(
            1e-9,
            second.width
            * second.height,
        )

        smaller_area = min(
            first_area,
            second_area,
        )

        overlap_ratio = (
            intersection_area
            / smaller_area
        )

        return (
            overlap_ratio
            >= 0.20
        )

    @staticmethod
    def _parse_box(
        value: Any,
    ) -> Optional[
        BoundingBox
    ]:

        if value is None:
            return None

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

    # ========================================================================
    # DIMENSIONS
    # ========================================================================

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
    # RESULT MANAGEMENT
    # ========================================================================

    def _store_result(
        self,
        result: SceneUnderstandingResult,
    ) -> None:

        with self._lock:
            self._last_result = result

    def get_last_result(
        self,
    ) -> Optional[
        SceneUnderstandingResult
    ]:

        with self._lock:
            return self._last_result


# ============================================================================
# SHARED ENGINE
# ============================================================================


_default_engine: Optional[
    SceneUnderstandingEngine
] = None

_default_engine_lock = (
    threading.RLock()
)


def get_scene_engine(
) -> SceneUnderstandingEngine:

    global _default_engine

    with _default_engine_lock:

        if _default_engine is None:

            _default_engine = (
                SceneUnderstandingEngine()
            )

        return _default_engine


# ============================================================================
# CONVENIENCE API
# ============================================================================


def understand_scene(
    image: Any = None,
    *,
    object_detections: Optional[
        Sequence[ObjectDetection]
    ] = None,
    pose_detections: Optional[
        Sequence[PoseDetection]
    ] = None,
    face_detections: Optional[
        Sequence[Any]
    ] = None,
    person_detections: Optional[
        Sequence[Any]
    ] = None,
    ocr_results: Optional[
        Sequence[Any]
    ] = None,
    screen_information: Optional[
        dict[str, Any]
    ] = None,
    document_information: Optional[
        dict[str, Any]
    ] = None,
) -> SceneUnderstandingResult:

    return get_scene_engine().understand(
        image,
        object_detections=object_detections,
        pose_detections=pose_detections,
        face_detections=face_detections,
        person_detections=person_detections,
        ocr_results=ocr_results,
        screen_information=screen_information,
        document_information=document_information,
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "SceneType",
    "SubjectType",
    "SpatialRelation",
    "SceneSubject",
    "SceneRelation",
    "SceneText",
    "SceneDescription",
    "SceneUnderstandingResult",
    "SceneUnderstandingConfig",
    "SceneUnderstandingEngine",
    "get_scene_engine",
    "understand_scene",
]


