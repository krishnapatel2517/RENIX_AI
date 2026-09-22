"""
RENIX Vision — Face Recognition

Provides the recognition layer used by RENIX after face detection.

Responsibilities:
- Register known face identities
- Store face embeddings in memory
- Generate embeddings from detected face crops
- Compare faces
- Recognize people from images
- Manage recognition thresholds
- Support multiple embedding backends
- Graceful fallback when optional ML libraries are unavailable
- Thread-safe identity management

Important:
This module separates FACE DETECTION from FACE RECOGNITION.

Detection answers:
    "Is there a face here?"

Recognition answers:
    "Does this face match a registered identity?"

The default lightweight backend uses image-based feature
signatures and is intended as a dependency-light fallback.
A stronger embedding backend can be plugged in later without
changing the public RENIX recognition API.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

from .face_detection import (
    BoundingBox,
    FaceDetection,
    FaceDetector,
    FaceDetectionResult,
    get_face_detector,
)

logger = logging.getLogger(
    "RENIX.vision.face_recognition"
)


# ============================================================================
# DATA TYPES
# ============================================================================


@dataclass
class FaceEmbedding:
    """
    Represents a numerical representation of a face.
    """

    vector: Any

    dimension: int

    backend: str

    created_at: float = field(
        default_factory=time.time
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_list(self) -> list[float]:
        """Return the embedding as a normal Python list."""

        if self.vector is None:
            return []

        try:
            return [
                float(value)
                for value in self.vector
            ]
        except Exception:
            return []

    def to_dict(
        self,
        include_vector: bool = False,
    ) -> dict[str, Any]:

        data = {
            "dimension": self.dimension,
            "backend": self.backend,
            "created_at": self.created_at,
            "metadata": dict(
                self.metadata
            ),
        }

        if include_vector:
            data["vector"] = self.to_list()

        return data


@dataclass
class FaceIdentity:
    """
    Registered person identity.
    """

    identity_id: str

    name: str

    embeddings: list[
        FaceEmbedding
    ] = field(default_factory=list)

    aliases: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: float = field(
        default_factory=time.time
    )

    updated_at: float = field(
        default_factory=time.time
    )

    enabled: bool = True

    def add_embedding(
        self,
        embedding: FaceEmbedding,
    ) -> None:

        self.embeddings.append(
            embedding
        )

        self.updated_at = time.time()

    def to_dict(
        self,
        include_embeddings: bool = False,
    ) -> dict[str, Any]:

        result = {
            "identity_id": self.identity_id,
            "name": self.name,
            "aliases": list(
                self.aliases
            ),
            "embedding_count": len(
                self.embeddings
            ),
            "metadata": dict(
                self.metadata
            ),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "enabled": self.enabled,
        }

        if include_embeddings:
            result["embeddings"] = [
                embedding.to_dict(
                    include_vector=True
                )
                for embedding
                in self.embeddings
            ]

        return result


@dataclass
class FaceMatch:
    """
    Recognition result for one detected face.
    """

    identity_id: Optional[str]

    name: Optional[str]

    similarity: float

    distance: float

    confidence: float

    recognized: bool

    face: Optional[
        FaceDetection
    ] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def unknown(self) -> bool:
        return not self.recognized

    def to_dict(self) -> dict[str, Any]:

        return {
            "identity_id": self.identity_id,
            "name": self.name,
            "similarity": self.similarity,
            "distance": self.distance,
            "confidence": self.confidence,
            "recognized": self.recognized,
            "unknown": self.unknown,
            "face": (
                self.face.to_dict()
                if self.face
                else None
            ),
            "metadata": dict(
                self.metadata
            ),
        }


@dataclass
class FaceRecognitionResult:
    """
    Complete recognition result for an image.
    """

    matches: list[
        FaceMatch
    ]

    timestamp: float

    processing_time: float

    backend: str

    success: bool = True

    error: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def recognized_count(self) -> int:

        return sum(
            1
            for match in self.matches
            if match.recognized
        )

    @property
    def unknown_count(self) -> int:

        return sum(
            1
            for match in self.matches
            if not match.recognized
        )

    @property
    def count(self) -> int:

        return len(self.matches)

    def recognized_names(
        self,
    ) -> list[str]:

        return [
            match.name
            for match in self.matches
            if match.recognized
            and match.name
        ]

    def to_dict(self) -> dict[str, Any]:

        return {
            "matches": [
                match.to_dict()
                for match in self.matches
            ],
            "count": self.count,
            "recognized_count": (
                self.recognized_count
            ),
            "unknown_count": (
                self.unknown_count
            ),
            "recognized_names": (
                self.recognized_names()
            ),
            "timestamp": self.timestamp,
            "processing_time": (
                self.processing_time
            ),
            "backend": self.backend,
            "success": self.success,
            "error": self.error,
            "metadata": dict(
                self.metadata
            ),
        }


@dataclass
class FaceRecognitionConfig:
    """
    Recognition configuration.
    """

    backend: str = "auto"

    threshold: float = 0.72

    unknown_threshold: float = 0.60

    max_identities: int = 1000

    max_embeddings_per_identity: int = 20

    normalize_embeddings: bool = True

    use_detection: bool = True

    detect_multiple_faces: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# FACE RECOGNIZER
# ============================================================================


class FaceRecognizer:
    """
    RENIX face recognition engine.

    Example
    -------

    ::

        recognizer = FaceRecognizer()

        recognizer.register_face(
            "krishna",
            "Krishna",
            image
        )

        result = recognizer.recognize(image)

        for match in result.matches:
            print(match.name, match.confidence)
    """

    def __init__(
        self,
        config: Optional[
            FaceRecognitionConfig
        ] = None,
        detector: Optional[
            FaceDetector
        ] = None,
    ) -> None:

        self.config = (
            config
            or FaceRecognitionConfig()
        )

        self._lock = threading.RLock()

        self._identities: dict[
            str,
            FaceIdentity,
        ] = {}

        self._detector = (
            detector
            or get_face_detector()
        )

        self._backend = (
            "numpy_signature"
            if NUMPY_AVAILABLE
            else "python_signature"
        )

        self._initialized = True

        self._recognition_calls = 0

        self._total_matches = 0

        self._total_recognized = 0

        self._total_processing_time = 0.0

        self._last_result: Optional[
            FaceRecognitionResult
        ] = None

    # ========================================================================
    # AVAILABILITY
    # ========================================================================

    def is_available(self) -> bool:
        """
        Return whether the recognition engine is available.

        The lightweight signature backend works without a
        specialized face-recognition package.
        """

        return self._initialized

    @property
    def backend(self) -> str:
        return self._backend

    # ========================================================================
    # IDENTITY MANAGEMENT
    # ========================================================================

    def register_identity(
        self,
        identity_id: str,
        name: str,
        *,
        aliases: Optional[
            Iterable[str]
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> FaceIdentity:
        """
        Create a new identity.
        """

        identity_id = str(
            identity_id
        ).strip()

        name = str(
            name
        ).strip()

        if not identity_id:
            raise ValueError(
                "identity_id cannot be empty."
            )

        if not name:
            raise ValueError(
                "name cannot be empty."
            )

        with self._lock:

            if (
                identity_id
                in self._identities
            ):

                raise ValueError(
                    f"Identity '{identity_id}' "
                    "already exists."
                )

            if (
                len(self._identities)
                >= self.config.max_identities
            ):

                raise RuntimeError(
                    "Maximum number of face "
                    "identities reached."
                )

            identity = FaceIdentity(
                identity_id=identity_id,
                name=name,
                aliases=list(
                    aliases or []
                ),
                metadata=dict(
                    metadata or {}
                ),
            )

            self._identities[
                identity_id
            ] = identity

            return identity

    def register_face(
        self,
        identity_id: str,
        name: str,
        image: Any,
        *,
        aliases: Optional[
            Iterable[str]
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> FaceIdentity:
        """
        Register an identity and create an embedding
        from an image.

        If the identity already exists, the new embedding
        is added to that identity.
        """

        with self._lock:

            identity = self._identities.get(
                identity_id
            )

        if identity is None:

            identity = self.register_identity(
                identity_id,
                name,
                aliases=aliases,
                metadata=metadata,
            )

        embedding = self.create_embedding(
            image
        )

        if embedding is None:

            raise ValueError(
                "Unable to create a face "
                "embedding from the supplied image."
            )

        self.add_embedding(
            identity_id,
            embedding,
        )

        return identity

    def add_embedding(
        self,
        identity_id: str,
        embedding: FaceEmbedding,
    ) -> bool:
        """
        Add an embedding to an existing identity.
        """

        with self._lock:

            identity = self._identities.get(
                identity_id
            )

            if identity is None:
                return False

            if (
                len(identity.embeddings)
                >= self.config.max_embeddings_per_identity
            ):

                # Keep the newest examples while
                # preventing unlimited memory growth.
                identity.embeddings.pop(
                    0
                )

            identity.add_embedding(
                embedding
            )

            return True

    def unregister_identity(
        self,
        identity_id: str,
    ) -> bool:
        """
        Remove a registered identity.
        """

        with self._lock:

            if (
                identity_id
                not in self._identities
            ):
                return False

            del self._identities[
                identity_id
            ]

            return True

    def enable_identity(
        self,
        identity_id: str,
    ) -> bool:

        with self._lock:

            identity = self._identities.get(
                identity_id
            )

            if identity is None:
                return False

            identity.enabled = True

            identity.updated_at = (
                time.time()
            )

            return True

    def disable_identity(
        self,
        identity_id: str,
    ) -> bool:

        with self._lock:

            identity = self._identities.get(
                identity_id
            )

            if identity is None:
                return False

            identity.enabled = False

            identity.updated_at = (
                time.time()
            )

            return True

    def get_identity(
        self,
        identity_id: str,
    ) -> Optional[FaceIdentity]:

        with self._lock:

            return self._identities.get(
                identity_id
            )

    def list_identities(
        self,
        *,
        include_disabled: bool = True,
    ) -> list[FaceIdentity]:

        with self._lock:

            identities = list(
                self._identities.values()
            )

            if not include_disabled:

                identities = [
                    identity
                    for identity in identities
                    if identity.enabled
                ]

            return identities

    def identity_count(self) -> int:

        with self._lock:

            return len(
                self._identities
            )

    # ========================================================================
    # EMBEDDING GENERATION
    # ========================================================================

    def create_embedding(
        self,
        image: Any,
        *,
        face: Optional[
            FaceDetection
        ] = None,
    ) -> Optional[FaceEmbedding]:
        """
        Create a lightweight numerical face representation.

        If a detected face is provided, its bounding box is
        cropped first. Otherwise the supplied image is used.
        """

        if image is None:
            return None

        try:

            face_image = image

            if face is not None:

                face_image = (
                    self._crop_face(
                        image,
                        face.box,
                    )
                )

                if face_image is None:
                    return None

            elif self.config.use_detection:

                detected = (
                    self._detector.detect(
                        image
                    )
                )

                if detected.faces:

                    largest = max(
                        detected.faces,
                        key=lambda item:
                        item.box.area,
                    )

                    face_image = (
                        self._crop_face(
                            image,
                            largest.box,
                        )
                    )

                    if face_image is None:
                        return None

            vector = (
                self._build_signature(
                    face_image
                )
            )

            if vector is None:
                return None

            if (
                self.config.normalize_embeddings
            ):

                vector = (
                    self._normalize(
                        vector
                    )
                )

            dimension = len(vector)

            return FaceEmbedding(
                vector=vector,
                dimension=dimension,
                backend=self._backend,
                metadata={
                    "generator": (
                        "RENIX"
                    ),
                },
            )

        except Exception:

            logger.exception(
                "Failed to create face embedding."
            )

            return None

    # ========================================================================
    # SIGNATURE BACKEND
    # ========================================================================

    def _build_signature(
        self,
        image: Any,
    ) -> Optional[list[float]]:
        """
        Generate a deterministic visual signature.

        This is a lightweight fallback and NOT a replacement
        for a neural face embedding model.
        """

        if image is None:
            return None

        if not NUMPY_AVAILABLE:

            return self._python_signature(
                image
            )

        try:

            array = np.asarray(
                image
            )

            if array.size == 0:
                return None

            if (
                array.ndim == 3
                and array.shape[2] >= 3
            ):

                # Convert BGR -> grayscale.
                if CV2_AVAILABLE:

                    gray = cv2.cvtColor(
                        array,
                        cv2.COLOR_BGR2GRAY,
                    )

                else:

                    gray = (
                        array[:, :, 0]
                        * 0.114
                        + array[:, :, 1]
                        * 0.587
                        + array[:, :, 2]
                        * 0.299
                    )

            elif array.ndim == 2:

                gray = array

            else:

                gray = np.squeeze(
                    array
                )

            gray = gray.astype(
                np.float32
            )

            # Resize to a stable low-dimensional representation.
            if CV2_AVAILABLE:

                gray = cv2.resize(
                    gray,
                    (32, 32),
                    interpolation=cv2.INTER_AREA,
                )

            else:

                gray = self._resize_numpy(
                    gray,
                    32,
                    32,
                )

            # Normalize intensity.
            mean = float(
                np.mean(gray)
            )

            std = float(
                np.std(gray)
            )

            if std > 1e-8:

                normalized = (
                    gray - mean
                ) / std

            else:

                normalized = (
                    gray - mean
                )

            # Low-resolution appearance vector.
            vector = (
                normalized.flatten()
            )

            # Add compact statistical features.
            statistics = np.array(
                [
                    float(
                        np.mean(gray)
                    ),
                    float(
                        np.std(gray)
                    ),
                    float(
                        np.min(gray)
                    ),
                    float(
                        np.max(gray)
                    ),
                    float(
                        np.percentile(
                            gray,
                            25,
                        )
                    ),
                    float(
                        np.percentile(
                            gray,
                            50,
                        )
                    ),
                    float(
                        np.percentile(
                            gray,
                            75,
                        )
                    ),
                ],
                dtype=np.float32,
            )

            vector = np.concatenate(
                [
                    vector,
                    statistics,
                ]
            )

            return [
                float(value)
                for value in vector
            ]

        except Exception:

            logger.exception(
                "Numpy signature generation failed."
            )

            return None

    def _python_signature(
        self,
        image: Any,
    ) -> Optional[list[float]]:
        """
        Dependency-light fallback.

        Uses object hashing only when NumPy is unavailable.
        This is suitable for API compatibility but is not
        a high-quality facial embedding.
        """

        try:

            raw = repr(
                image
            ).encode(
                "utf-8",
                errors="ignore",
            )

            digest = hashlib.sha256(
                raw
            ).digest()

            return [
                byte / 255.0
                for byte in digest
            ]

        except Exception:

            return None

    @staticmethod
    def _resize_numpy(
        image: Any,
        width: int,
        height: int,
    ) -> Any:

        if not NUMPY_AVAILABLE:
            return image

        source_height, source_width = (
            image.shape[:2]
        )

        if (
            source_width == width
            and source_height == height
        ):
            return image

        y_indices = (
            np.linspace(
                0,
                source_height - 1,
                height,
            ).astype(int)
        )

        x_indices = (
            np.linspace(
                0,
                source_width - 1,
                width,
            ).astype(int)
        )

        return image[
            np.ix_(
                y_indices,
                x_indices,
            )
        ]

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize(
        vector: Iterable[float],
    ) -> list[float]:

        values = [
            float(value)
            for value in vector
        ]

        norm = math.sqrt(
            sum(
                value * value
                for value in values
            )
        )

        if norm <= 1e-12:

            return values

        return [
            value / norm
            for value in values
        ]

    # ========================================================================
    # DISTANCE / SIMILARITY
    # ========================================================================

    def compare_embeddings(
        self,
        first: FaceEmbedding,
        second: FaceEmbedding,
    ) -> tuple[
        float,
        float,
    ]:
        """
        Compare two embeddings.

        Returns:
            (similarity, distance)
        """

        a = first.to_list()

        b = second.to_list()

        if not a or not b:

            return 0.0, float("inf")

        if len(a) != len(b):

            raise ValueError(
                "Embedding dimensions do not match."
            )

        dot = sum(
            x * y
            for x, y in zip(
                a,
                b,
            )
        )

        norm_a = math.sqrt(
            sum(
                x * x
                for x in a
            )
        )

        norm_b = math.sqrt(
            sum(
                y * y
                for y in b
            )
        )

        if (
            norm_a <= 1e-12
            or norm_b <= 1e-12
        ):

            return 0.0, float("inf")

        cosine = (
            dot
            / (
                norm_a
                * norm_b
            )
        )

        cosine = max(
            -1.0,
            min(
                1.0,
                cosine,
            ),
        )

        similarity = (
            cosine + 1.0
        ) / 2.0

        distance = 1.0 - similarity

        return (
            float(similarity),
            float(distance),
        )

    def compare(
        self,
        first_image: Any,
        second_image: Any,
    ) -> float:
        """
        Compare two face images.

        Returns similarity in the range 0..1.
        """

        first = (
            self.create_embedding(
                first_image
            )
        )

        second = (
            self.create_embedding(
                second_image
            )
        )

        if (
            first is None
            or second is None
        ):
            return 0.0

        similarity, _ = (
            self.compare_embeddings(
                first,
                second,
            )
        )

        return similarity

    # ========================================================================
    # RECOGNITION
    # ========================================================================

    def recognize(
        self,
        image: Any,
    ) -> FaceRecognitionResult:
        """
        Recognize all faces in an image.
        """

        started = time.perf_counter()

        timestamp = time.time()

        if image is None:

            return self._failure_result(
                timestamp,
                started,
                "Input image is None.",
            )

        try:

            if self.config.use_detection:

                detection_result = (
                    self._detector.detect(
                        image
                    )
                )

                if not detection_result.success:

                    return self._failure_result(
                        timestamp,
                        started,
                        (
                            detection_result.error
                            or
                            "Face detection failed."
                        ),
                    )

                faces = (
                    detection_result.faces
                )

            else:

                faces = []

                embedding = (
                    self.create_embedding(
                        image
                    )
                )

                if embedding is None:

                    return self._failure_result(
                        timestamp,
                        started,
                        "Unable to create "
                        "face embedding.",
                    )

                match = (
                    self._match_embedding(
                        embedding
                    )
                )

                matches = [
                    FaceMatch(
                        identity_id=(
                            match["identity_id"]
                        ),
                        name=(
                            match["name"]
                        ),
                        similarity=(
                            match["similarity"]
                        ),
                        distance=(
                            match["distance"]
                        ),
                        confidence=(
                            match["confidence"]
                        ),
                        recognized=(
                            match["recognized"]
                        ),
                        face=None,
                    )
                ]

                return self._success_result(
                    timestamp,
                    started,
                    matches,
                )

            matches: list[
                FaceMatch
            ] = []

            for face in faces:

                face_image = (
                    self._crop_face(
                        image,
                        face.box,
                    )
                )

                if face_image is None:

                    matches.append(
                        FaceMatch(
                            identity_id=None,
                            name=None,
                            similarity=0.0,
                            distance=1.0,
                            confidence=0.0,
                            recognized=False,
                            face=face,
                            metadata={
                                "error":
                                "Unable to crop face."
                            },
                        )
                    )

                    continue

                embedding = (
                    self.create_embedding(
                        face_image,
                        face=None,
                    )
                )

                if embedding is None:

                    matches.append(
                        FaceMatch(
                            identity_id=None,
                            name=None,
                            similarity=0.0,
                            distance=1.0,
                            confidence=0.0,
                            recognized=False,
                            face=face,
                        )
                    )

                    continue

                best = (
                    self._match_embedding(
                        embedding
                    )
                )

                matches.append(
                    FaceMatch(
                        identity_id=(
                            best["identity_id"]
                        ),
                        name=(
                            best["name"]
                        ),
                        similarity=(
                            best["similarity"]
                        ),
                        distance=(
                            best["distance"]
                        ),
                        confidence=(
                            best["confidence"]
                        ),
                        recognized=(
                            best["recognized"]
                        ),
                        face=face,
                        metadata={
                            "backend": (
                                self._backend
                            ),
                        },
                    )
                )

                if not (
                    self.config.detect_multiple_faces
                ):
                    break

            return self._success_result(
                timestamp,
                started,
                matches,
            )

        except Exception as exc:

            logger.exception(
                "Face recognition failed."
            )

            return self._failure_result(
                timestamp,
                started,
                str(exc),
            )

    def recognize_face(
        self,
        image: Any,
    ) -> Optional[FaceMatch]:
        """
        Recognize the most likely face in an image.
        """

        result = self.recognize(
            image
        )

        if not result.matches:
            return None

        return max(
            result.matches,
            key=lambda match:
            match.confidence,
        )

    # ========================================================================
    # MATCHING
    # ========================================================================

    def _match_embedding(
        self,
        embedding: FaceEmbedding,
    ) -> dict[str, Any]:
        """
        Find the best identity for an embedding.
        """

        best_identity: Optional[
            FaceIdentity
        ] = None

        best_similarity = 0.0

        best_distance = 1.0

        with self._lock:

            identities = [
                identity
                for identity
                in self._identities.values()
                if identity.enabled
            ]

        for identity in identities:

            for known_embedding in (
                identity.embeddings
            ):

                try:

                    similarity, distance = (
                        self.compare_embeddings(
                            embedding,
                            known_embedding,
                        )
                    )

                except ValueError:

                    continue

                if (
                    similarity
                    > best_similarity
                ):

                    best_similarity = (
                        similarity
                    )

                    best_distance = (
                        distance
                    )

                    best_identity = (
                        identity
                    )

        recognized = (
            best_identity is not None
            and best_similarity
            >= self.config.threshold
        )

        if (
            best_identity is None
            or best_similarity
            < self.config.unknown_threshold
        ):

            recognized = False

            identity_id = None

            name = None

        elif recognized:

            identity_id = (
                best_identity.identity_id
            )

            name = (
                best_identity.name
            )

        else:

            identity_id = None

            name = None

        confidence = (
            best_similarity
            if recognized
            else max(
                0.0,
                min(
                    1.0,
                    best_similarity
                    * 0.8,
                ),
            )
        )

        return {
            "identity_id": identity_id,
            "name": name,
            "similarity": best_similarity,
            "distance": best_distance,
            "confidence": confidence,
            "recognized": recognized,
        }

    # ========================================================================
    # CROPPING
    # ========================================================================

    @staticmethod
    def _crop_face(
        image: Any,
        box: BoundingBox,
    ) -> Any:

        try:

            return image[
                box.y : box.y2,
                box.x : box.x2,
            ].copy()

        except Exception:

            logger.debug(
                "Face crop failed.",
                exc_info=True,
            )

            return None

    # ========================================================================
    # RESULT HELPERS
    # ========================================================================

    def _success_result(
        self,
        timestamp: float,
        started: float,
        matches: list[FaceMatch],
    ) -> FaceRecognitionResult:

        processing_time = (
            time.perf_counter()
            - started
        )

        result = FaceRecognitionResult(
            matches=matches,
            timestamp=timestamp,
            processing_time=processing_time,
            backend=self._backend,
            success=True,
        )

        self._record_result(
            result
        )

        return result

    def _failure_result(
        self,
        timestamp: float,
        started: float,
        error: str,
    ) -> FaceRecognitionResult:

        processing_time = (
            time.perf_counter()
            - started
        )

        result = FaceRecognitionResult(
            matches=[],
            timestamp=timestamp,
            processing_time=processing_time,
            backend=self._backend,
            success=False,
            error=error,
        )

        self._record_result(
            result
        )

        return result

    def _record_result(
        self,
        result: FaceRecognitionResult,
    ) -> None:

        with self._lock:

            self._last_result = result

            self._recognition_calls += 1

            self._total_matches += (
                result.count
            )

            self._total_recognized += (
                result.recognized_count
            )

            self._total_processing_time += (
                result.processing_time
            )

    # ========================================================================
    # LAST RESULT
    # ========================================================================

    def get_last_result(
        self,
    ) -> Optional[FaceRecognitionResult]:

        with self._lock:

            return self._last_result

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def get_statistics(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            calls = (
                self._recognition_calls
            )

            average_time = (
                self._total_processing_time
                / calls
                if calls > 0
                else 0.0
            )

            return {
                "backend": self._backend,
                "available": self.is_available(),
                "identity_count": (
                    len(self._identities)
                ),
                "recognition_calls": calls,
                "total_faces": (
                    self._total_matches
                ),
                "total_recognized": (
                    self._total_recognized
                ),
                "average_processing_time": (
                    average_time
                ),
                "threshold": (
                    self.config.threshold
                ),
                "unknown_threshold": (
                    self.config.unknown_threshold
                ),
            }

    def reset_statistics(
        self,
    ) -> None:

        with self._lock:

            self._recognition_calls = 0

            self._total_matches = 0

            self._total_recognized = 0

            self._total_processing_time = 0.0

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    def set_threshold(
        self,
        threshold: float,
    ) -> None:

        threshold = float(
            threshold
        )

        if not 0.0 <= threshold <= 1.0:

            raise ValueError(
                "threshold must be between 0 and 1."
            )

        with self._lock:

            self.config.threshold = (
                threshold
            )

    def set_unknown_threshold(
        self,
        threshold: float,
    ) -> None:

        threshold = float(
            threshold
        )

        if not 0.0 <= threshold <= 1.0:

            raise ValueError(
                "unknown threshold must "
                "be between 0 and 1."
            )

        with self._lock:

            self.config.unknown_threshold = (
                threshold
            )

    # ========================================================================
    # RESET
    # ========================================================================

    def clear_identities(
        self,
    ) -> None:

        with self._lock:

            self._identities.clear()

    def reset(
        self,
    ) -> None:

        with self._lock:

            self._identities.clear()

            self.reset_statistics()

            self._last_result = None


# ============================================================================
# DEFAULT RECOGNIZER
# ============================================================================


_default_recognizer: Optional[
    FaceRecognizer
] = None

_default_recognizer_lock = (
    threading.RLock()
)


def get_face_recognizer() -> FaceRecognizer:
    """Return the shared RENIX face recognizer."""

    global _default_recognizer

    with _default_recognizer_lock:

        if _default_recognizer is None:

            _default_recognizer = (
                FaceRecognizer()
            )

        return _default_recognizer


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def recognize(
    image: Any,
) -> FaceRecognitionResult:
    """Recognize faces using the default recognizer."""

    return get_face_recognizer().recognize(
        image
    )


def recognize_face(
    image: Any,
) -> Optional[FaceMatch]:
    """Recognize the strongest face match."""

    return get_face_recognizer().recognize_face(
        image
    )


def register_face(
    identity_id: str,
    name: str,
    image: Any,
) -> FaceIdentity:
    """Register a face with the default recognizer."""

    return get_face_recognizer().register_face(
        identity_id,
        name,
        image,
    )


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "FaceEmbedding",
    "FaceIdentity",
    "FaceMatch",
    "FaceRecognitionResult",
    "FaceRecognitionConfig",
    "FaceRecognizer",
    "get_face_recognizer",
    "recognize",
    "recognize_face",
    "register_face",
    "CV2_AVAILABLE",
    "NUMPY_AVAILABLE",
]


