"""
RENIX AI - Voice Authentication

Provides voice-based authentication for RENIX.

Responsibilities:
- Create voice enrollment profiles
- Store voice feature templates
- Compare incoming voice samples against enrolled profiles
- Calculate similarity/confidence
- Authenticate a speaker
- Lock/unlock RENIX voice access
- Manage multiple authorized voice profiles
- Thread-safe profile management
- Persistent JSON storage
- Optional integration with an external speaker-verification backend

Important:
This module provides an application-level voice-authentication layer.
It should NOT be treated as a cryptographic identity system by itself.
For high-security operations, RENIX should combine voice authentication
with another factor such as face authentication, PIN, or explicit confirmation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger("RENIX.voice.voice_authentication")


# ============================================================================
# DATA TYPES
# ============================================================================


@dataclass
class VoiceAuthConfig:
    """Configuration for voice authentication."""

    storage_path: str = (
        "data/profiles/voice_profiles.json"
    )

    similarity_threshold: float = 0.82

    high_confidence_threshold: float = 0.92

    max_failed_attempts: int = 5

    lockout_seconds: float = 30.0

    enrollment_samples: int = 3

    min_sample_length: int = 100

    normalize_audio: bool = True

    require_enrollment: bool = True

    enabled: bool = True

    # If True, the detector can use a registered backend.
    backend_enabled: bool = False


@dataclass
class VoiceFeature:
    """
    Extracted voice representation.

    This intentionally stores statistical/acoustic features rather than
    raw microphone recordings.
    """

    rms: float

    peak: float

    zero_crossing_rate: float

    mean_abs: float

    variance: float

    spectral_centroid: float = 0.0

    spectral_bandwidth: float = 0.0

    spectral_rolloff: float = 0.0

    duration: float = 0.0

    sample_rate: int = 16000

    feature_version: str = "1.0"

    def to_vector(self) -> list[float]:
        return [
            self.rms,
            self.peak,
            self.zero_crossing_rate,
            self.mean_abs,
            self.variance,
            self.spectral_centroid,
            self.spectral_bandwidth,
            self.spectral_rolloff,
            self.duration,
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VoiceProfile:
    """Registered RENIX voice profile."""

    profile_id: str

    display_name: str

    feature_templates: list[VoiceFeature] = field(
        default_factory=list
    )

    created_at: float = field(
        default_factory=time.time
    )

    updated_at: float = field(
        default_factory=time.time
    )

    enabled: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:

        return {
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "feature_templates": [
                feature.to_dict()
                for feature in self.feature_templates
            ],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "VoiceProfile":

        features = []

        for item in data.get(
            "feature_templates",
            [],
        ):

            features.append(
                VoiceFeature(
                    **item
                )
            )

        return cls(
            profile_id=str(
                data["profile_id"]
            ),
            display_name=str(
                data["display_name"]
            ),
            feature_templates=features,
            created_at=float(
                data.get(
                    "created_at",
                    time.time(),
                )
            ),
            updated_at=float(
                data.get(
                    "updated_at",
                    time.time(),
                )
            ),
            enabled=bool(
                data.get(
                    "enabled",
                    True,
                )
            ),
            metadata=dict(
                data.get(
                    "metadata",
                    {},
                )
            ),
        )


@dataclass
class VoiceAuthenticationResult:
    """Result returned after speaker verification."""

    authenticated: bool

    profile_id: Optional[str] = None

    display_name: Optional[str] = None

    confidence: float = 0.0

    similarity: float = 0.0

    method: str = "local_features"

    reason: str = ""

    timestamp: float = field(
        default_factory=time.time
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:

        return {
            "authenticated": self.authenticated,
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "confidence": self.confidence,
            "similarity": self.similarity,
            "method": self.method,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class VoiceAuthenticationError(RuntimeError):
    """Base voice-authentication exception."""


class VoiceAuthenticationLocked(
    VoiceAuthenticationError
):
    """Raised when authentication is temporarily locked."""


# ============================================================================
# STORAGE
# ============================================================================


class VoiceProfileStore:
    """Persistent JSON storage for voice profiles."""

    def __init__(
        self,
        path: str,
    ) -> None:

        self.path = Path(
            path
        )

        self._lock = threading.RLock()

        self._profiles: dict[
            str,
            VoiceProfile,
        ] = {}

        self._ensure_directory()

        self.load()

    def _ensure_directory(self) -> None:

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load(self) -> None:

        with self._lock:

            if not self.path.exists():

                self._profiles = {}

                return

            try:

                raw = self.path.read_text(
                    encoding="utf-8"
                )

                data = json.loads(
                    raw
                )

                profiles = {}

                for item in data.get(
                    "profiles",
                    [],
                ):

                    profile = VoiceProfile.from_dict(
                        item
                    )

                    profiles[
                        profile.profile_id
                    ] = profile

                self._profiles = profiles

            except Exception as exc:

                logger.exception(
                    "Failed loading voice profiles."
                )

                raise VoiceAuthenticationError(
                    "Could not load voice profiles."
                ) from exc

    def save(self) -> None:

        with self._lock:

            self._ensure_directory()

            data = {
                "version": 1,
                "updated_at": time.time(),
                "profiles": [
                    profile.to_dict()
                    for profile in self._profiles.values()
                ],
            }

            serialized = json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            )

            # Atomic write.
            fd, temp_path = tempfile.mkstemp(
                prefix="voice_profiles_",
                suffix=".tmp",
                dir=str(
                    self.path.parent
                ),
            )

            try:

                with os.fdopen(
                    fd,
                    "w",
                    encoding="utf-8",
                ) as file:

                    file.write(
                        serialized
                    )

                    file.flush()

                    os.fsync(
                        file.fileno()
                    )

                os.replace(
                    temp_path,
                    self.path,
                )

            finally:

                if os.path.exists(
                    temp_path
                ):

                    try:

                        os.remove(
                            temp_path
                        )

                    except OSError:

                        pass

    def add(
        self,
        profile: VoiceProfile,
    ) -> None:

        with self._lock:

            self._profiles[
                profile.profile_id
            ] = profile

            self.save()

    def get(
        self,
        profile_id: str,
    ) -> Optional[VoiceProfile]:

        with self._lock:

            return self._profiles.get(
                profile_id
            )

    def remove(
        self,
        profile_id: str,
    ) -> bool:

        with self._lock:

            if profile_id not in self._profiles:

                return False

            del self._profiles[
                profile_id
            ]

            self.save()

            return True

    def all(
        self,
    ) -> list[VoiceProfile]:

        with self._lock:

            return list(
                self._profiles.values()
            )

    def clear(self) -> None:

        with self._lock:

            self._profiles.clear()

            self.save()


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================


class VoiceFeatureExtractor:
    """
    Extracts lightweight acoustic features.

    The extractor does not require an ML model and therefore provides a
    dependable baseline when a dedicated speaker-embedding model is absent.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
    ) -> None:

        self.sample_rate = (
            sample_rate
        )

    def extract(
        self,
        samples: Any,
        *,
        sample_rate: Optional[int] = None,
    ) -> VoiceFeature:

        values = self._flatten(
            samples
        )

        if len(values) < 2:

            raise VoiceAuthenticationError(
                "Voice sample is too short."
            )

        rate = int(
            sample_rate
            or self.sample_rate
        )

        if rate <= 0:

            raise ValueError(
                "sample_rate must be positive."
            )

        if self._contains_nan_or_inf(
            values
        ):

            raise VoiceAuthenticationError(
                "Voice sample contains invalid values."
            )

        rms = self._rms(
            values
        )

        peak = self._peak(
            values
        )

        mean_abs = (
            sum(
                abs(value)
                for value in values
            )
            / len(values)
        )

        mean = (
            sum(values)
            / len(values)
        )

        variance = (
            sum(
                (value - mean) ** 2
                for value in values
            )
            / len(values)
        )

        zero_crossing_rate = (
            self._zero_crossing_rate(
                values
            )
        )

        spectral_centroid = 0.0

        spectral_bandwidth = 0.0

        spectral_rolloff = 0.0

        try:

            spectral_centroid = (
                self._spectral_centroid(
                    values,
                    rate,
                )
            )

            spectral_bandwidth = (
                self._spectral_bandwidth(
                    values,
                    rate,
                )
            )

            spectral_rolloff = (
                self._spectral_rolloff(
                    values,
                    rate,
                )
            )

        except Exception:

            logger.debug(
                "Spectral feature extraction failed.",
                exc_info=True,
            )

        duration = (
            len(values)
            / rate
        )

        return VoiceFeature(
            rms=rms,
            peak=peak,
            zero_crossing_rate=(
                zero_crossing_rate
            ),
            mean_abs=mean_abs,
            variance=variance,
            spectral_centroid=(
                spectral_centroid
            ),
            spectral_bandwidth=(
                spectral_bandwidth
            ),
            spectral_rolloff=(
                spectral_rolloff
            ),
            duration=duration,
            sample_rate=rate,
        )

    @staticmethod
    def _flatten(
        samples: Any,
    ) -> list[float]:

        if hasattr(
            samples,
            "flatten",
        ):

            return [
                float(value)
                for value in samples.flatten()
            ]

        values = []

        for item in samples:

            if isinstance(
                item,
                (list, tuple),
            ):

                values.extend(
                    float(value)
                    for value in item
                )

            else:

                values.append(
                    float(item)
                )

        return values

    @staticmethod
    def _contains_nan_or_inf(
        values: Iterable[float],
    ) -> bool:

        return any(
            not math.isfinite(value)
            for value in values
        )

    @staticmethod
    def _rms(
        values: list[float],
    ) -> float:

        return math.sqrt(
            sum(
                value * value
                for value in values
            )
            / len(values)
        )

    @staticmethod
    def _peak(
        values: list[float],
    ) -> float:

        return max(
            abs(value)
            for value in values
        )

    @staticmethod
    def _zero_crossing_rate(
        values: list[float],
    ) -> float:

        crossings = 0

        for index in range(
            1,
            len(values),
        ):

            previous = values[
                index - 1
            ]

            current = values[
                index
            ]

            if (
                previous < 0
                and current >= 0
            ) or (
                previous >= 0
                and current < 0
            ):

                crossings += 1

        return (
            crossings
            / max(
                1,
                len(values) - 1,
            )
        )

    @staticmethod
    def _fft_magnitude(
        values: list[float],
    ) -> list[float]:

        """
        Lightweight DFT fallback.

        For production systems, RENIX can replace this with NumPy FFT.
        """

        n = len(values)

        # Keep the fallback computationally reasonable.
        max_points = 512

        if n > max_points:

            step = n / max_points

            values = [
                values[
                    min(
                        n - 1,
                        int(
                            index * step
                        ),
                    )
                ]
                for index in range(
                    max_points
                )
            ]

            n = len(values)

        magnitudes = []

        half = n // 2

        for k in range(
            half
        ):

            real = 0.0

            imag = 0.0

            for index, value in enumerate(
                values
            ):

                angle = (
                    2
                    * math.pi
                    * k
                    * index
                    / n
                )

                real += (
                    value
                    * math.cos(angle)
                )

                imag -= (
                    value
                    * math.sin(angle)
                )

            magnitudes.append(
                math.sqrt(
                    real * real
                    + imag * imag
                )
            )

        return magnitudes

    def _spectral_centroid(
        self,
        values: list[float],
        sample_rate: int,
    ) -> float:

        magnitudes = (
            self._fft_magnitude(
                values
            )
        )

        if not magnitudes:

            return 0.0

        weighted = 0.0

        total = 0.0

        n = len(values)

        for index, magnitude in enumerate(
            magnitudes
        ):

            frequency = (
                index
                * sample_rate
                / n
            )

            weighted += (
                frequency
                * magnitude
            )

            total += magnitude

        if total <= 0:

            return 0.0

        return weighted / total

    def _spectral_bandwidth(
        self,
        values: list[float],
        sample_rate: int,
    ) -> float:

        magnitudes = (
            self._fft_magnitude(
                values
            )
        )

        if not magnitudes:

            return 0.0

        centroid = (
            self._spectral_centroid(
                values,
                sample_rate,
            )
        )

        weighted = 0.0

        total = 0.0

        n = len(values)

        for index, magnitude in enumerate(
            magnitudes
        ):

            frequency = (
                index
                * sample_rate
                / n
            )

            weighted += (
                ((frequency - centroid) ** 2)
                * magnitude
            )

            total += magnitude

        if total <= 0:

            return 0.0

        return math.sqrt(
            weighted / total
        )

    def _spectral_rolloff(
        self,
        values: list[float],
        sample_rate: int,
        rolloff: float = 0.85,
    ) -> float:

        magnitudes = (
            self._fft_magnitude(
                values
            )
        )

        if not magnitudes:

            return 0.0

        total = sum(
            magnitudes
        )

        target = (
            total
            * rolloff
        )

        accumulated = 0.0

        n = len(values)

        for index, magnitude in enumerate(
            magnitudes
        ):

            accumulated += magnitude

            if accumulated >= target:

                return (
                    index
                    * sample_rate
                    / n
                )

        return sample_rate / 2


# ============================================================================
# AUTHENTICATION ENGINE
# ============================================================================


class VoiceAuthentication:
    """
    RENIX voice-authentication manager.
    """

    def __init__(
        self,
        config: Optional[
            VoiceAuthConfig
        ] = None,
        *,
        feature_extractor: Optional[
            VoiceFeatureExtractor
        ] = None,
    ) -> None:

        self.config = (
            config
            or VoiceAuthConfig()
        )

        self._lock = threading.RLock()

        self.store = VoiceProfileStore(
            self.config.storage_path
        )

        self.extractor = (
            feature_extractor
            or VoiceFeatureExtractor()
        )

        self._failed_attempts = 0

        self._locked_until = 0.0

        self._last_result: Optional[
            VoiceAuthenticationResult
        ] = None

        self._callbacks: list[
            Callable[
                [VoiceAuthenticationResult],
                None,
            ]
        ] = []

        self._validate_config()

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def _validate_config(self) -> None:

        if not (
            0.0
            <= self.config.similarity_threshold
            <= 1.0
        ):

            raise ValueError(
                "similarity_threshold must be between 0 and 1."
            )

        if not (
            0.0
            <= self.config.high_confidence_threshold
            <= 1.0
        ):

            raise ValueError(
                "high_confidence_threshold must be between 0 and 1."
            )

        if (
            self.config.max_failed_attempts
            <= 0
        ):

            raise ValueError(
                "max_failed_attempts must be positive."
            )

        if self.config.lockout_seconds < 0:

            raise ValueError(
                "lockout_seconds cannot be negative."
            )

    # =========================================================================
    # PROFILE MANAGEMENT
    # =========================================================================

    def create_profile_id(
        self,
        display_name: str,
    ) -> str:

        normalized = re.sub(
            r"\s+",
            " ",
            display_name.strip().lower(),
        )

        if not normalized:

            raise ValueError(
                "display_name cannot be empty."
            )

        digest = hashlib.sha256(
            normalized.encode(
                "utf-8"
            )
        ).hexdigest()[:16]

        return (
            "voice_"
            + digest
        )

    def enroll(
        self,
        display_name: str,
        samples: Iterable[Any],
        *,
        sample_rate: int = 16000,
        profile_id: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> VoiceProfile:

        if not self.config.enabled:

            raise VoiceAuthenticationError(
                "Voice authentication is disabled."
            )

        display_name = str(
            display_name
        ).strip()

        if not display_name:

            raise ValueError(
                "display_name cannot be empty."
            )

        sample_list = list(
            samples
        )

        if len(sample_list) < (
            self.config.enrollment_samples
        ):

            raise VoiceAuthenticationError(
                f"At least "
                f"{self.config.enrollment_samples} "
                f"voice samples are required."
            )

        features: list[
            VoiceFeature
        ] = []

        for sample in sample_list:

            feature = (
                self.extractor.extract(
                    sample,
                    sample_rate=sample_rate,
                )
            )

            features.append(
                feature
            )

        profile_id = (
            profile_id
            or self.create_profile_id(
                display_name
            )
        )

        profile = VoiceProfile(
            profile_id=profile_id,
            display_name=display_name,
            feature_templates=features,
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
        )

        self.store.add(
            profile
        )

        return profile

    def add_sample(
        self,
        profile_id: str,
        sample: Any,
        *,
        sample_rate: int = 16000,
    ) -> VoiceProfile:

        profile = self.store.get(
            profile_id
        )

        if profile is None:

            raise VoiceAuthenticationError(
                f"Unknown voice profile: {profile_id}"
            )

        feature = (
            self.extractor.extract(
                sample,
                sample_rate=sample_rate,
            )
        )

        profile.feature_templates.append(
            feature
        )

        profile.updated_at = time.time()

        self.store.add(
            profile
        )

        return profile

    def delete_profile(
        self,
        profile_id: str,
    ) -> bool:

        return self.store.remove(
            profile_id
        )

    def get_profile(
        self,
        profile_id: str,
    ) -> Optional[VoiceProfile]:

        return self.store.get(
            profile_id
        )

    def list_profiles(
        self,
    ) -> list[VoiceProfile]:

        return self.store.all()

    def enable_profile(
        self,
        profile_id: str,
    ) -> bool:

        profile = self.store.get(
            profile_id
        )

        if profile is None:

            return False

        profile.enabled = True

        profile.updated_at = time.time()

        self.store.add(
            profile
        )

        return True

    def disable_profile(
        self,
        profile_id: str,
    ) -> bool:

        profile = self.store.get(
            profile_id
        )

        if profile is None:

            return False

        profile.enabled = False

        profile.updated_at = time.time()

        self.store.add(
            profile
        )

        return True

    # =========================================================================
    # VERIFICATION
    # =========================================================================

    def authenticate(
        self,
        sample: Any,
        *,
        sample_rate: int = 16000,
        profile_id: Optional[str] = None,
    ) -> VoiceAuthenticationResult:

        if not self.config.enabled:

            return self._result(
                authenticated=False,
                reason="voice_authentication_disabled",
            )

        if self.is_locked():

            return self._result(
                authenticated=False,
                reason="temporarily_locked",
            )

        profiles = self.list_profiles()

        if profile_id is not None:

            profiles = [
                profile
                for profile in profiles
                if profile.profile_id
                == profile_id
            ]

        profiles = [
            profile
            for profile in profiles
            if profile.enabled
            and profile.feature_templates
        ]

        if not profiles:

            return self._result(
                authenticated=False,
                reason="no_voice_profiles_available",
            )

        try:

            feature = (
                self.extractor.extract(
                    sample,
                    sample_rate=sample_rate,
                )
            )

        except Exception as exc:

            return self._result(
                authenticated=False,
                reason=(
                    "voice_feature_extraction_failed"
                ),
                metadata={
                    "error": str(exc)
                },
            )

        best_profile: Optional[
            VoiceProfile
        ] = None

        best_similarity = 0.0

        for profile in profiles:

            similarity = (
                self._profile_similarity(
                    feature,
                    profile,
                )
            )

            if similarity > best_similarity:

                best_similarity = similarity

                best_profile = profile

        if best_profile is None:

            return self._record_failure(
                reason="no_matching_profile"
            )

        authenticated = (
            best_similarity
            >= self.config.similarity_threshold
        )

        if authenticated:

            self._failed_attempts = 0

            confidence = (
                self._confidence_from_similarity(
                    best_similarity
                )
            )

            result = self._result(
                authenticated=True,
                profile_id=(
                    best_profile.profile_id
                ),
                display_name=(
                    best_profile.display_name
                ),
                confidence=confidence,
                similarity=best_similarity,
                reason="voice_verified",
            )

            self._notify(
                result
            )

            return result

        return self._record_failure(
            reason="voice_not_verified",
            similarity=best_similarity,
            metadata={
                "best_profile_id": (
                    best_profile.profile_id
                ),
            },
        )

    def _profile_similarity(
        self,
        feature: VoiceFeature,
        profile: VoiceProfile,
    ) -> float:

        similarities = []

        for template in (
            profile.feature_templates
        ):

            similarities.append(
                self._feature_similarity(
                    feature,
                    template,
                )
            )

        if not similarities:

            return 0.0

        # Use the strongest template match.
        return max(
            similarities
        )

    def _feature_similarity(
        self,
        first: VoiceFeature,
        second: VoiceFeature,
    ) -> float:

        a = first.to_vector()

        b = second.to_vector()

        if len(a) != len(b):

            return 0.0

        # Feature-specific tolerances.
        scales = [
            0.25,       # RMS
            0.35,       # peak
            0.10,       # ZCR
            0.25,       # mean abs
            0.20,       # variance
            3000.0,     # centroid
            4000.0,     # bandwidth
            4000.0,     # rolloff
            2.0,        # duration
        ]

        scores = []

        for x, y, scale in zip(
            a,
            b,
            scales,
        ):

            denominator = max(
                scale,
                abs(y),
                1e-9,
            )

            difference = abs(
                x - y
            )

            normalized_difference = (
                difference
                / denominator
            )

            score = max(
                0.0,
                1.0
                - normalized_difference,
            )

            scores.append(
                score
            )

        if not scores:

            return 0.0

        # Acoustic characteristics that are relatively stable across
        # utterances receive greater influence.
        weights = [
            0.10,
            0.05,
            0.15,
            0.10,
            0.10,
            0.20,
            0.10,
            0.10,
            0.10,
        ]

        weighted_score = sum(
            score * weight
            for score, weight in zip(
                scores,
                weights,
            )
        )

        total_weight = sum(
            weights
        )

        return max(
            0.0,
            min(
                1.0,
                weighted_score
                / total_weight,
            ),
        )

    @staticmethod
    def _confidence_from_similarity(
        similarity: float,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                similarity,
            ),
        )

    # =========================================================================
    # FAILURE / LOCKOUT
    # =========================================================================

    def _record_failure(
        self,
        *,
        reason: str,
        similarity: float = 0.0,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> VoiceAuthenticationResult:

        with self._lock:

            self._failed_attempts += 1

            if (
                self._failed_attempts
                >= self.config.max_failed_attempts
            ):

                self._locked_until = (
                    time.monotonic()
                    + self.config.lockout_seconds
                )

                reason = (
                    "maximum_failed_attempts"
                )

        result = self._result(
            authenticated=False,
            confidence=(
                self._confidence_from_similarity(
                    similarity
                )
            ),
            similarity=similarity,
            reason=reason,
            metadata=(
                metadata
                or {}
            ),
        )

        self._notify(
            result
        )

        return result

    def is_locked(self) -> bool:

        with self._lock:

            if (
                self._locked_until <= 0
            ):

                return False

            if (
                time.monotonic()
                >= self._locked_until
            ):

                self._locked_until = 0.0

                self._failed_attempts = 0

                return False

            return True

    def unlock(
        self,
    ) -> None:

        with self._lock:

            self._locked_until = 0.0

            self._failed_attempts = 0

    def remaining_lockout(
        self,
    ) -> float:

        with self._lock:

            remaining = (
                self._locked_until
                - time.monotonic()
            )

            return max(
                0.0,
                remaining,
            )

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def add_callback(
        self,
        callback: Callable[
            [VoiceAuthenticationResult],
            None,
        ],
    ) -> None:

        if not callable(callback):

            raise ValueError(
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
            [VoiceAuthenticationResult],
            None,
        ],
    ) -> bool:

        with self._lock:

            if callback in self._callbacks:

                self._callbacks.remove(
                    callback
                )

                return True

        return False

    def clear_callbacks(
        self,
    ) -> None:

        with self._lock:

            self._callbacks.clear()

    def _notify(
        self,
        result: VoiceAuthenticationResult,
    ) -> None:

        self._last_result = result

        with self._lock:

            callbacks = list(
                self._callbacks
            )

        for callback in callbacks:

            try:

                callback(
                    result
                )

            except Exception:

                logger.exception(
                    "Voice authentication callback failed."
                )

    # =========================================================================
    # RESULT
    # =========================================================================

    def _result(
        self,
        *,
        authenticated: bool,
        profile_id: Optional[str] = None,
        display_name: Optional[str] = None,
        confidence: float = 0.0,
        similarity: float = 0.0,
        reason: str = "",
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> VoiceAuthenticationResult:

        result = VoiceAuthenticationResult(
            authenticated=authenticated,
            profile_id=profile_id,
            display_name=display_name,
            confidence=confidence,
            similarity=similarity,
            reason=reason,
            metadata=(
                metadata
                or {}
            ),
        )

        self._last_result = result

        return result

    def get_last_result(
        self,
    ) -> Optional[
        VoiceAuthenticationResult
    ]:

        return self._last_result

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(
        self,
    ) -> dict[str, Any]:

        return {
            "enabled": self.config.enabled,
            "profiles": len(
                self.list_profiles()
            ),
            "failed_attempts": (
                self._failed_attempts
            ),
            "locked": self.is_locked(),
            "remaining_lockout": (
                self.remaining_lockout()
            ),
            "similarity_threshold": (
                self.config.similarity_threshold
            ),
            "high_confidence_threshold": (
                self.config.high_confidence_threshold
            ),
            "storage_path": (
                self.config.storage_path
            ),
            "last_result": (
                self._last_result.to_dict()
                if self._last_result
                else None
            ),
        }

    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    def shutdown(
        self,
    ) -> None:

        self.clear_callbacks()

        self.unlock()

        self._last_result = None


# ============================================================================
# ALIASES
# ============================================================================


VoiceAuthenticator = VoiceAuthentication


__all__ = [
    "VoiceAuthConfig",
    "VoiceFeature",
    "VoiceProfile",
    "VoiceAuthenticationResult",
    "VoiceAuthenticationError",
    "VoiceAuthenticationLocked",
    "VoiceProfileStore",
    "VoiceFeatureExtractor",
    "VoiceAuthentication",
    "VoiceAuthenticator",
]


