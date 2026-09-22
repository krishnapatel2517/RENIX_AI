"""
RENIX AI - Speaker Identification

Identifies which enrolled voice profile most closely matches an incoming
voice sample.

This module is intentionally separate from voice_authentication.py:

    speaker_identification.py
        -> "WHO is speaking?"

    voice_authentication.py
        -> "IS this speaker authorized?"

Typical RENIX pipeline:

    Microphone
        ↓
    Speech-to-Text / Audio Capture
        ↓
    SpeakerIdentification
        ↓
    VoiceAuthentication
        ↓
    RENIX User Context / Security
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from .voice_authentication import (
    VoiceFeature,
    VoiceFeatureExtractor,
    VoiceProfile,
    VoiceProfileStore,
)

logger = logging.getLogger(
    "RENIX.voice.speaker_identification"
)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class SpeakerIdentificationConfig:
    """Configuration for speaker identification."""

    storage_path: str = (
        "data/profiles/voice_profiles.json"
    )

    identification_threshold: float = 0.60

    high_confidence_threshold: float = 0.85

    sample_rate: int = 16000

    max_candidates: int = 5

    enabled: bool = True

    use_multiple_templates: bool = True

    # Number of strongest template matches used to calculate
    # the final profile score.
    top_template_count: int = 3


# ============================================================================
# RESULT TYPES
# ============================================================================


@dataclass
class SpeakerCandidate:
    """A possible speaker."""

    profile_id: str

    display_name: str

    similarity: float

    confidence: float

    template_count: int

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:

        return {
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "similarity": self.similarity,
            "confidence": self.confidence,
            "template_count": self.template_count,
            "metadata": self.metadata,
        }


@dataclass
class SpeakerIdentificationResult:
    """Final speaker-identification result."""

    identified: bool

    profile_id: Optional[str] = None

    display_name: Optional[str] = None

    confidence: float = 0.0

    similarity: float = 0.0

    candidates: list[SpeakerCandidate] = field(
        default_factory=list
    )

    reason: str = ""

    timestamp: float = field(
        default_factory=time.time
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:

        return {
            "identified": self.identified,
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "confidence": self.confidence,
            "similarity": self.similarity,
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "reason": self.reason,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class SpeakerIdentificationError(
    RuntimeError
):
    """Base speaker-identification error."""


# ============================================================================
# SPEAKER IDENTIFIER
# ============================================================================


class SpeakerIdentifier:
    """
    RENIX speaker-identification engine.

    It compares a new voice sample against all enabled enrolled profiles
    and returns ranked candidates.

    Example:

        identifier = SpeakerIdentifier()

        result = identifier.identify(
            microphone_samples
        )

        if result.identified:
            print(
                "Speaker:",
                result.display_name
            )
    """

    def __init__(
        self,
        config: Optional[
            SpeakerIdentificationConfig
        ] = None,
        *,
        profile_store: Optional[
            VoiceProfileStore
        ] = None,
        feature_extractor: Optional[
            VoiceFeatureExtractor
        ] = None,
    ) -> None:

        self.config = (
            config
            or SpeakerIdentificationConfig()
        )

        self._lock = threading.RLock()

        self.store = (
            profile_store
            or VoiceProfileStore(
                self.config.storage_path
            )
        )

        self.extractor = (
            feature_extractor
            or VoiceFeatureExtractor(
                self.config.sample_rate
            )
        )

        self._last_result: Optional[
            SpeakerIdentificationResult
        ] = None

        self._callbacks: list[
            Callable[
                [SpeakerIdentificationResult],
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
            <= self.config.identification_threshold
            <= 1.0
        ):

            raise ValueError(
                "identification_threshold must be "
                "between 0 and 1."
            )

        if not (
            0.0
            <= self.config.high_confidence_threshold
            <= 1.0
        ):

            raise ValueError(
                "high_confidence_threshold must be "
                "between 0 and 1."
            )

        if self.config.sample_rate <= 0:

            raise ValueError(
                "sample_rate must be positive."
            )

        if self.config.max_candidates <= 0:

            raise ValueError(
                "max_candidates must be positive."
            )

        if self.config.top_template_count <= 0:

            raise ValueError(
                "top_template_count must be positive."
            )

    def configure(
        self,
        *,
        identification_threshold: Optional[
            float
        ] = None,
        high_confidence_threshold: Optional[
            float
        ] = None,
        max_candidates: Optional[int] = None,
        top_template_count: Optional[
            int
        ] = None,
        enabled: Optional[bool] = None,
    ) -> SpeakerIdentificationConfig:

        if identification_threshold is not None:

            self.config.identification_threshold = (
                float(
                    identification_threshold
                )
            )

        if high_confidence_threshold is not None:

            self.config.high_confidence_threshold = (
                float(
                    high_confidence_threshold
                )
            )

        if max_candidates is not None:

            self.config.max_candidates = int(
                max_candidates
            )

        if top_template_count is not None:

            self.config.top_template_count = int(
                top_template_count
            )

        if enabled is not None:

            self.config.enabled = bool(
                enabled
            )

        self._validate_config()

        return self.config

    # =========================================================================
    # IDENTIFICATION
    # =========================================================================

    def identify(
        self,
        samples: Any,
        *,
        sample_rate: Optional[int] = None,
        candidate_profile_ids: Optional[
            Iterable[str]
        ] = None,
    ) -> SpeakerIdentificationResult:
        """
        Identify the most likely enrolled speaker.
        """

        if not self.config.enabled:

            return self._make_result(
                identified=False,
                reason="speaker_identification_disabled",
            )

        profiles = self._get_profiles(
            candidate_profile_ids
        )

        if not profiles:

            return self._make_result(
                identified=False,
                reason="no_enabled_profiles",
            )

        try:

            feature = (
                self.extractor.extract(
                    samples,
                    sample_rate=(
                        sample_rate
                        or self.config.sample_rate
                    ),
                )
            )

        except Exception as exc:

            return self._make_result(
                identified=False,
                reason="feature_extraction_failed",
                metadata={
                    "error": str(exc)
                },
            )

        candidates: list[
            SpeakerCandidate
        ] = []

        for profile in profiles:

            try:

                similarity = (
                    self._profile_similarity(
                        feature,
                        profile,
                    )
                )

                confidence = (
                    self._similarity_to_confidence(
                        similarity
                    )
                )

                candidates.append(
                    SpeakerCandidate(
                        profile_id=(
                            profile.profile_id
                        ),
                        display_name=(
                            profile.display_name
                        ),
                        similarity=similarity,
                        confidence=confidence,
                        template_count=len(
                            profile.feature_templates
                        ),
                    )
                )

            except Exception:

                logger.exception(
                    "Failed comparing profile %s.",
                    profile.profile_id,
                )

        candidates.sort(
            key=lambda candidate: (
                candidate.similarity
            ),
            reverse=True,
        )

        candidates = candidates[
            : self.config.max_candidates
        ]

        if not candidates:

            return self._make_result(
                identified=False,
                reason="no_comparable_profiles",
            )

        best = candidates[0]

        identified = (
            best.similarity
            >= self.config.identification_threshold
        )

        if identified:

            result = self._make_result(
                identified=True,
                profile_id=(
                    best.profile_id
                ),
                display_name=(
                    best.display_name
                ),
                confidence=(
                    best.confidence
                ),
                similarity=(
                    best.similarity
                ),
                candidates=candidates,
                reason="speaker_identified",
            )

        else:

            result = self._make_result(
                identified=False,
                confidence=(
                    best.confidence
                ),
                similarity=(
                    best.similarity
                ),
                candidates=candidates,
                reason="speaker_not_identified",
            )

        self._notify(
            result
        )

        return result

    # =========================================================================
    # PROFILE RETRIEVAL
    # =========================================================================

    def _get_profiles(
        self,
        candidate_profile_ids: Optional[
            Iterable[str]
        ],
    ) -> list[VoiceProfile]:

        profiles = [
            profile
            for profile in self.store.all()
            if profile.enabled
            and profile.feature_templates
        ]

        if candidate_profile_ids is None:

            return profiles

        allowed = {
            str(profile_id)
            for profile_id
            in candidate_profile_ids
        }

        return [
            profile
            for profile in profiles
            if profile.profile_id
            in allowed
        ]

    # =========================================================================
    # FEATURE COMPARISON
    # =========================================================================

    def _profile_similarity(
        self,
        feature: VoiceFeature,
        profile: VoiceProfile,
    ) -> float:

        template_scores = []

        for template in (
            profile.feature_templates
        ):

            score = (
                self._feature_similarity(
                    feature,
                    template,
                )
            )

            template_scores.append(
                score
            )

        if not template_scores:

            return 0.0

        template_scores.sort(
            reverse=True
        )

        if self.config.use_multiple_templates:

            count = min(
                self.config.top_template_count,
                len(template_scores),
            )

            selected = template_scores[
                :count
            ]

            # Strongest sample gets slightly more influence.
            if len(selected) == 1:

                return selected[0]

            weights = [
                1.0 / (
                    index + 1
                )
                for index in range(
                    len(selected)
                )
            ]

            weighted = sum(
                score * weight
                for score, weight in zip(
                    selected,
                    weights,
                )
            )

            weight_total = sum(
                weights
            )

            return (
                weighted
                / weight_total
            )

        return max(
            template_scores
        )

    def _feature_similarity(
        self,
        first: VoiceFeature,
        second: VoiceFeature,
    ) -> float:

        # Each value is normalized against a sensible acoustic scale.
        scales = [
            0.25,       # RMS
            0.35,       # peak
            0.10,       # zero crossing rate
            0.25,       # mean absolute amplitude
            0.20,       # variance
            3000.0,     # spectral centroid
            4000.0,     # spectral bandwidth
            4000.0,     # spectral rolloff
            2.0,        # duration
        ]

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

        first_vector = (
            first.to_vector()
        )

        second_vector = (
            second.to_vector()
        )

        if len(first_vector) != len(
            second_vector
        ):

            return 0.0

        scores = []

        for (
            value_a,
            value_b,
            scale,
        ) in zip(
            first_vector,
            second_vector,
            scales,
        ):

            difference = abs(
                value_a
                - value_b
            )

            denominator = max(
                scale,
                abs(value_b),
                1e-9,
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
    def _similarity_to_confidence(
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
    # TOP SPEAKER
    # =========================================================================

    def identify_top(
        self,
        samples: Any,
        *,
        sample_rate: Optional[int] = None,
    ) -> Optional[SpeakerCandidate]:

        result = self.identify(
            samples,
            sample_rate=sample_rate,
        )

        if not result.candidates:

            return None

        return result.candidates[0]

    # =========================================================================
    # PROFILE-SPECIFIC VERIFICATION
    # =========================================================================

    def matches_profile(
        self,
        samples: Any,
        profile_id: str,
        *,
        sample_rate: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> bool:

        result = self.identify(
            samples,
            sample_rate=sample_rate,
            candidate_profile_ids=[
                profile_id
            ],
        )

        required = (
            threshold
            if threshold is not None
            else self.config.identification_threshold
        )

        return (
            result.identified
            and result.profile_id
            == profile_id
            and result.similarity
            >= required
        )

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def add_callback(
        self,
        callback: Callable[
            [SpeakerIdentificationResult],
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
            [SpeakerIdentificationResult],
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
        result: SpeakerIdentificationResult,
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
                    "Speaker identification callback failed."
                )

    # =========================================================================
    # LAST RESULT / STATUS
    # =========================================================================

    def get_last_result(
        self,
    ) -> Optional[
        SpeakerIdentificationResult
    ]:

        return self._last_result

    def get_status(
        self,
    ) -> dict[str, Any]:

        profiles = self.store.all()

        enabled_profiles = [
            profile
            for profile in profiles
            if profile.enabled
        ]

        return {
            "enabled": self.config.enabled,
            "profile_count": len(
                profiles
            ),
            "enabled_profile_count": len(
                enabled_profiles
            ),
            "identification_threshold": (
                self.config.identification_threshold
            ),
            "high_confidence_threshold": (
                self.config.high_confidence_threshold
            ),
            "max_candidates": (
                self.config.max_candidates
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

        self._last_result = None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def identify_speaker(
    samples: Any,
    *,
    storage_path: str = (
        "data/profiles/voice_profiles.json"
    ),
    sample_rate: int = 16000,
) -> SpeakerIdentificationResult:
    """
    Convenience function for one-shot speaker identification.
    """

    identifier = SpeakerIdentifier(
        SpeakerIdentificationConfig(
            storage_path=storage_path,
            sample_rate=sample_rate,
        )
    )

    try:

        return identifier.identify(
            samples,
            sample_rate=sample_rate,
        )

    finally:

        identifier.shutdown()


# ============================================================================
# ALIASES
# ============================================================================


SpeakerIdentification = SpeakerIdentifier


__all__ = [
    "SpeakerIdentificationConfig",
    "SpeakerCandidate",
    "SpeakerIdentificationResult",
    "SpeakerIdentificationError",
    "SpeakerIdentifier",
    "SpeakerIdentification",
    "identify_speaker",
]


