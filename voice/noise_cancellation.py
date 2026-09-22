"""
RENIX AI - Noise Cancellation

Audio preprocessing layer for RENIX voice systems.

Responsibilities:
- Remove DC offset
- Normalize microphone samples
- Suppress low-level background noise
- Apply a simple noise gate
- Reduce stationary noise
- Reduce low-frequency rumble
- Reduce high-frequency hiss
- Preserve speech as much as possible
- Provide streaming-friendly processing
- Work without requiring heavy external dependencies

Pipeline:

    Microphone
        ↓
    NoiseCancellation
        ↓
    Speech-to-Text
    Speaker Identification
    Voice Authentication
    Voice Activity Detection

The implementation is intentionally dependency-light.

If NumPy is installed, RENIX can pass NumPy arrays directly.
Plain Python lists and tuples are also supported.
"""

from __future__ import annotations

import logging
import math
import statistics
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Optional

logger = logging.getLogger(
    "RENIX.voice.noise_cancellation"
)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class NoiseCancellationConfig:
    """Configuration for RENIX noise cancellation."""

    sample_rate: int = 16000

    channels: int = 1

    enabled: bool = True

    normalize: bool = True

    remove_dc: bool = True

    noise_gate_enabled: bool = True

    noise_gate_threshold: float = 0.015

    noise_gate_ratio: float = 0.15

    noise_gate_attack: float = 0.01

    noise_gate_release: float = 0.10

    noise_reduction_enabled: bool = True

    noise_reduction_strength: float = 0.65

    noise_estimation_frames: int = 10

    low_cut_enabled: bool = True

    low_cut_hz: float = 70.0

    high_cut_enabled: bool = True

    high_cut_hz: float = 7600.0

    output_peak: float = 0.95

    preserve_length: bool = True


# ============================================================================
# RESULT TYPES
# ============================================================================


@dataclass
class NoiseProfile:
    """
    Estimated stationary background-noise characteristics.
    """

    rms: float = 0.0

    peak: float = 0.0

    mean: float = 0.0

    variance: float = 0.0

    zero_crossing_rate: float = 0.0

    created_at: float = 0.0

    sample_count: int = 0

    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:

        if self.metadata is None:

            self.metadata = {}


@dataclass
class NoiseCancellationResult:
    """
    Output of noise-cancellation processing.
    """

    samples: Any

    input_rms: float

    output_rms: float

    input_peak: float

    output_peak: float

    reduction_ratio: float

    noise_profile_used: bool

    samples_processed: int

    sample_rate: int

    metadata: dict[str, Any]

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "input_rms": self.input_rms,
            "output_rms": self.output_rms,
            "input_peak": self.input_peak,
            "output_peak": self.output_peak,
            "reduction_ratio": self.reduction_ratio,
            "noise_profile_used": (
                self.noise_profile_used
            ),
            "samples_processed": (
                self.samples_processed
            ),
            "sample_rate": self.sample_rate,
            "metadata": self.metadata,
        }


class NoiseCancellationError(
    RuntimeError
):
    """Base noise-cancellation error."""


# ============================================================================
# BASIC AUDIO UTILITIES
# ============================================================================


class AudioMath:
    """Dependency-free audio calculations."""

    @staticmethod
    def flatten(
        samples: Any,
    ) -> list[float]:

        if samples is None:

            return []

        if hasattr(
            samples,
            "flatten",
        ):

            return [
                float(value)
                for value in samples.flatten()
            ]

        values: list[float] = []

        try:

            for item in samples:

                if isinstance(
                    item,
                    (list, tuple),
                ):

                    for value in item:

                        values.append(
                            float(value)
                        )

                else:

                    values.append(
                        float(item)
                    )

        except TypeError:

            values.append(
                float(samples)
            )

        return values

    @staticmethod
    def rms(
        samples: Iterable[float],
    ) -> float:

        values = list(
            samples
        )

        if not values:

            return 0.0

        return math.sqrt(
            sum(
                value * value
                for value in values
            )
            / len(values)
        )

    @staticmethod
    def peak(
        samples: Iterable[float],
    ) -> float:

        values = list(
            samples
        )

        if not values:

            return 0.0

        return max(
            abs(value)
            for value in values
        )

    @staticmethod
    def mean(
        samples: Iterable[float],
    ) -> float:

        values = list(
            samples
        )

        if not values:

            return 0.0

        return (
            sum(values)
            / len(values)
        )

    @staticmethod
    def variance(
        samples: Iterable[float],
    ) -> float:

        values = list(
            samples
        )

        if len(values) < 2:

            return 0.0

        return statistics.pvariance(
            values
        )

    @staticmethod
    def zero_crossing_rate(
        samples: Iterable[float],
    ) -> float:

        values = list(
            samples
        )

        if len(values) < 2:

            return 0.0

        crossings = 0

        for index in range(
            1,
            len(values),
        ):

            a = values[
                index - 1
            ]

            b = values[
                index
            ]

            if (
                a < 0
                and b >= 0
            ) or (
                a >= 0
                and b < 0
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
    def clamp(
        value: float,
        minimum: float = -1.0,
        maximum: float = 1.0,
    ) -> float:

        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )


# ============================================================================
# NOISE CANCELLATION ENGINE
# ============================================================================


class NoiseCancellation:
    """
    Main RENIX noise-cancellation engine.

    Example:

        config = NoiseCancellationConfig(
            sample_rate=16000
        )

        processor = NoiseCancellation(
            config
        )

        result = processor.process(
            microphone_samples
        )

        cleaned_audio = result.samples
    """

    def __init__(
        self,
        config: Optional[
            NoiseCancellationConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or NoiseCancellationConfig()
        )

        self._lock = threading.RLock()

        self._noise_profile: Optional[
            NoiseProfile
        ] = None

        self._previous_output: float = 0.0

        self._gate_gain: float = 1.0

        self._low_pass_state: float = 0.0

        self._high_pass_state: float = 0.0

        self._last_result: Optional[
            NoiseCancellationResult
        ] = None

        self._validate_config()

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def _validate_config(
        self,
    ) -> None:

        if self.config.sample_rate <= 0:

            raise ValueError(
                "sample_rate must be positive."
            )

        if self.config.channels <= 0:

            raise ValueError(
                "channels must be positive."
            )

        if not (
            0.0
            <= self.config.noise_gate_threshold
            <= 1.0
        ):

            raise ValueError(
                "noise_gate_threshold must "
                "be between 0 and 1."
            )

        if not (
            0.0
            <= self.config.noise_gate_ratio
            <= 1.0
        ):

            raise ValueError(
                "noise_gate_ratio must "
                "be between 0 and 1."
            )

        if not (
            0.0
            <= self.config.noise_reduction_strength
            <= 1.0
        ):

            raise ValueError(
                "noise_reduction_strength must "
                "be between 0 and 1."
            )

        if self.config.low_cut_hz < 0:

            raise ValueError(
                "low_cut_hz cannot be negative."
            )

        if self.config.high_cut_hz <= 0:

            raise ValueError(
                "high_cut_hz must be positive."
            )

        if (
            self.config.low_cut_hz
            >= self.config.high_cut_hz
        ):

            raise ValueError(
                "low_cut_hz must be below "
                "high_cut_hz."
            )

        if not (
            0.0
            < self.config.output_peak
            <= 1.0
        ):

            raise ValueError(
                "output_peak must be "
                "between 0 and 1."
            )

    # =========================================================================
    # PROCESS
    # =========================================================================

    def process(
        self,
        samples: Any,
        *,
        sample_rate: Optional[int] = None,
        estimate_noise: bool = False,
    ) -> NoiseCancellationResult:
        """
        Process an audio buffer.

        Args:
            samples:
                Audio samples. Can be a list, tuple, or NumPy array.

            sample_rate:
                Optional sample-rate override.

            estimate_noise:
                If True, the current buffer is treated as background-noise
                material and used to update the internal noise profile.
        """

        if not self.config.enabled:

            values = AudioMath.flatten(
                samples
            )

            return self._build_result(
                values,
                values.copy(),
                noise_profile_used=False,
            )

        values = AudioMath.flatten(
            samples
        )

        if not values:

            return self._build_result(
                [],
                [],
                noise_profile_used=False,
            )

        rate = int(
            sample_rate
            or self.config.sample_rate
        )

        if rate <= 0:

            raise NoiseCancellationError(
                "Invalid sample rate."
            )

        self._validate_samples(
            values
        )

        original = values.copy()

        if self.config.remove_dc:

            values = self.remove_dc(
                values
            )

        if estimate_noise:

            self.estimate_noise(
                values
            )

        if self.config.low_cut_enabled:

            values = self.high_pass_filter(
                values,
                rate,
                self.config.low_cut_hz,
            )

        if self.config.high_cut_enabled:

            values = self.low_pass_filter(
                values,
                rate,
                self.config.high_cut_hz,
            )

        if self.config.noise_reduction_enabled:

            values = self.reduce_stationary_noise(
                values
            )

        if self.config.noise_gate_enabled:

            values = self.noise_gate(
                values,
                rate,
            )

        if self.config.normalize:

            values = self.normalize(
                values,
                self.config.output_peak,
            )

        result = self._build_result(
            original,
            values,
            noise_profile_used=(
                self._noise_profile
                is not None
            ),
        )

        self._last_result = result

        return result

    # =========================================================================
    # SAMPLE VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_samples(
        samples: Iterable[float],
    ) -> None:

        for value in samples:

            if not math.isfinite(
                value
            ):

                raise NoiseCancellationError(
                    "Audio contains NaN or infinity."
                )

    # =========================================================================
    # DC REMOVAL
    # =========================================================================

    @staticmethod
    def remove_dc(
        samples: list[float],
    ) -> list[float]:

        if not samples:

            return []

        mean = AudioMath.mean(
            samples
        )

        return [
            value - mean
            for value in samples
        ]

    # =========================================================================
    # NOISE ESTIMATION
    # =========================================================================

    def estimate_noise(
        self,
        samples: Any,
    ) -> NoiseProfile:

        values = AudioMath.flatten(
            samples
        )

        if not values:

            raise NoiseCancellationError(
                "Cannot estimate noise from empty audio."
            )

        profile = NoiseProfile(
            rms=AudioMath.rms(
                values
            ),
            peak=AudioMath.peak(
                values
            ),
            mean=AudioMath.mean(
                values
            ),
            variance=AudioMath.variance(
                values
            ),
            zero_crossing_rate=(
                AudioMath.zero_crossing_rate(
                    values
                )
            ),
            created_at=__import__(
                "time"
            ).time(),
            sample_count=len(
                values
            ),
            metadata={
                "sample_rate": (
                    self.config.sample_rate
                )
            },
        )

        with self._lock:

            if self._noise_profile is None:

                self._noise_profile = profile

            else:

                self._noise_profile = (
                    self._merge_noise_profiles(
                        self._noise_profile,
                        profile,
                    )
                )

        return self._noise_profile

    @staticmethod
    def _merge_noise_profiles(
        previous: NoiseProfile,
        current: NoiseProfile,
    ) -> NoiseProfile:

        previous_count = max(
            1,
            previous.sample_count,
        )

        current_count = max(
            1,
            current.sample_count,
        )

        total = (
            previous_count
            + current_count
        )

        def weighted(
            first: float,
            second: float,
        ) -> float:

            return (
                first
                * previous_count
                + second
                * current_count
            ) / total

        return NoiseProfile(
            rms=weighted(
                previous.rms,
                current.rms,
            ),
            peak=weighted(
                previous.peak,
                current.peak,
            ),
            mean=weighted(
                previous.mean,
                current.mean,
            ),
            variance=weighted(
                previous.variance,
                current.variance,
            ),
            zero_crossing_rate=weighted(
                previous.zero_crossing_rate,
                current.zero_crossing_rate,
            ),
            created_at=current.created_at,
            sample_count=total,
            metadata={
                **(
                    previous.metadata
                    or {}
                ),
                **(
                    current.metadata
                    or {}
                ),
            },
        )

    def set_noise_profile(
        self,
        profile: NoiseProfile,
    ) -> None:

        if not isinstance(
            profile,
            NoiseProfile,
        ):

            raise TypeError(
                "profile must be a NoiseProfile."
            )

        with self._lock:

            self._noise_profile = profile

    def get_noise_profile(
        self,
    ) -> Optional[NoiseProfile]:

        with self._lock:

            return self._noise_profile

    def clear_noise_profile(
        self,
    ) -> None:

        with self._lock:

            self._noise_profile = None

    # =========================================================================
    # STATIONARY NOISE REDUCTION
    # =========================================================================

    def reduce_stationary_noise(
        self,
        samples: list[float],
    ) -> list[float]:

        if not samples:

            return []

        profile = self._noise_profile

        if profile is None:

            # No explicit noise model:
            # estimate the quieter part of the current buffer.
            sorted_values = sorted(
                abs(value)
                for value in samples
            )

            index = int(
                len(sorted_values)
                * 0.15
            )

            index = min(
                len(sorted_values) - 1,
                max(
                    0,
                    index,
                ),
            )

            estimated_noise = (
                sorted_values[index]
            )

        else:

            estimated_noise = (
                profile.rms
            )

        strength = (
            self.config.noise_reduction_strength
        )

        if estimated_noise <= 0:

            return samples.copy()

        output = []

        # Soft subtraction instead of hard clipping.
        for value in samples:

            magnitude = abs(
                value
            )

            sign = (
                1.0
                if value >= 0
                else -1.0
            )

            if magnitude <= estimated_noise:

                cleaned = (
                    magnitude
                    * (
                        1.0
                        - strength
                    )
                )

            else:

                reduction = (
                    estimated_noise
                    * strength
                )

                cleaned = max(
                    0.0,
                    magnitude
                    - reduction,
                )

            output.append(
                sign * cleaned
            )

        return output

    # =========================================================================
    # NOISE GATE
    # =========================================================================

    def noise_gate(
        self,
        samples: list[float],
        sample_rate: int,
    ) -> list[float]:

        if not samples:

            return []

        threshold = (
            self.config.noise_gate_threshold
        )

        floor = (
            self.config.noise_gate_ratio
        )

        attack_samples = max(
            1,
            int(
                self.config.noise_gate_attack
                * sample_rate
            ),
        )

        release_samples = max(
            1,
            int(
                self.config.noise_gate_release
                * sample_rate
            ),
        )

        output = []

        for value in samples:

            amplitude = abs(
                value
            )

            if amplitude >= threshold:

                target_gain = 1.0

                smoothing = (
                    1.0
                    / attack_samples
                )

            else:

                target_gain = floor

                smoothing = (
                    1.0
                    / release_samples
                )

            self._gate_gain += (
                target_gain
                - self._gate_gain
            ) * smoothing

            output.append(
                value
                * self._gate_gain
            )

        return output

    # =========================================================================
    # HIGH-PASS FILTER
    # =========================================================================

    def high_pass_filter(
        self,
        samples: list[float],
        sample_rate: int,
        cutoff_hz: float,
    ) -> list[float]:

        if not samples:

            return []

        if cutoff_hz <= 0:

            return samples.copy()

        dt = 1.0 / sample_rate

        rc = (
            1.0
            / (
                2.0
                * math.pi
                * cutoff_hz
            )
        )

        alpha = (
            rc
            / (
                rc
                + dt
            )
        )

        previous_input = (
            self._high_pass_state
        )

        previous_output = (
            0.0
        )

        output = []

        for value in samples:

            filtered = (
                alpha
                * (
                    previous_output
                    + value
                    - previous_input
                )
            )

            output.append(
                filtered
            )

            previous_input = value

            previous_output = filtered

        self._high_pass_state = (
            previous_input
        )

        return output

    # =========================================================================
    # LOW-PASS FILTER
    # =========================================================================

    def low_pass_filter(
        self,
        samples: list[float],
        sample_rate: int,
        cutoff_hz: float,
    ) -> list[float]:

        if not samples:

            return []

        if cutoff_hz <= 0:

            return samples.copy()

        dt = 1.0 / sample_rate

        rc = (
            1.0
            / (
                2.0
                * math.pi
                * cutoff_hz
            )
        )

        alpha = (
            dt
            / (
                rc
                + dt
            )
        )

        previous = (
            self._low_pass_state
        )

        output = []

        for value in samples:

            previous += (
                alpha
                * (
                    value
                    - previous
                )
            )

            output.append(
                previous
            )

        self._low_pass_state = (
            previous
        )

        return output

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    @staticmethod
    def normalize(
        samples: list[float],
        target_peak: float = 0.95,
    ) -> list[float]:

        if not samples:

            return []

        peak = AudioMath.peak(
            samples
        )

        if peak <= 1e-12:

            return samples.copy()

        gain = (
            target_peak
            / peak
        )

        return [
            AudioMath.clamp(
                value * gain
            )
            for value in samples
        ]

    # =========================================================================
    # ADAPTIVE PROCESSING
    # =========================================================================

    def process_adaptive(
        self,
        samples: Any,
        *,
        sample_rate: Optional[int] = None,
    ) -> NoiseCancellationResult:

        values = AudioMath.flatten(
            samples
        )

        if not values:

            return self.process(
                values,
                sample_rate=sample_rate,
            )

        # Estimate the quietest part of the buffer.
        frame_count = max(
            1,
            self.config.noise_estimation_frames,
        )

        frame_size = max(
            1,
            len(values)
            // frame_count,
        )

        frames = []

        for index in range(
            frame_count
        ):

            start = (
                index
                * frame_size
            )

            end = (
                start
                + frame_size
            )

            frame = values[
                start:end
            ]

            if frame:

                frames.append(
                    frame
                )

        if frames:

            frame_rms = [
                AudioMath.rms(
                    frame
                )
                for frame in frames
            ]

            quietest_index = min(
                range(
                    len(frames)
                ),
                key=lambda index: (
                    frame_rms[index]
                ),
            )

            self.estimate_noise(
                frames[
                    quietest_index
                ]
            )

        return self.process(
            values,
            sample_rate=sample_rate,
        )

    # =========================================================================
    # STREAM RESET
    # =========================================================================

    def reset_stream(
        self,
    ) -> None:

        with self._lock:

            self._previous_output = 0.0

            self._gate_gain = 1.0

            self._low_pass_state = 0.0

            self._high_pass_state = 0.0

            self._last_result = None

    # =========================================================================
    # RESULT
    # =========================================================================

    def _build_result(
        self,
        original: list[float],
        processed: list[float],
        *,
        noise_profile_used: bool,
    ) -> NoiseCancellationResult:

        input_rms = AudioMath.rms(
            original
        )

        output_rms = AudioMath.rms(
            processed
        )

        input_peak = AudioMath.peak(
            original
        )

        output_peak = AudioMath.peak(
            processed
        )

        if input_rms <= 1e-12:

            reduction_ratio = 0.0

        else:

            reduction_ratio = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - (
                        output_rms
                        / input_rms
                    ),
                ),
            )

        return NoiseCancellationResult(
            samples=processed,
            input_rms=input_rms,
            output_rms=output_rms,
            input_peak=input_peak,
            output_peak=output_peak,
            reduction_ratio=reduction_ratio,
            noise_profile_used=(
                noise_profile_used
            ),
            samples_processed=len(
                original
            ),
            sample_rate=(
                self.config.sample_rate
            ),
            metadata={
                "noise_gate_enabled": (
                    self.config.noise_gate_enabled
                ),
                "noise_reduction_enabled": (
                    self.config.noise_reduction_enabled
                ),
                "low_cut_enabled": (
                    self.config.low_cut_enabled
                ),
                "high_cut_enabled": (
                    self.config.high_cut_enabled
                ),
            },
        )

    def get_last_result(
        self,
    ) -> Optional[
        NoiseCancellationResult
    ]:

        return self._last_result

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(
        self,
    ) -> dict[str, Any]:

        profile = (
            self.get_noise_profile()
        )

        return {
            "enabled": self.config.enabled,
            "sample_rate": (
                self.config.sample_rate
            ),
            "channels": (
                self.config.channels
            ),
            "noise_profile_available": (
                profile is not None
            ),
            "noise_profile_rms": (
                profile.rms
                if profile
                else None
            ),
            "noise_gate_enabled": (
                self.config.noise_gate_enabled
            ),
            "noise_reduction_enabled": (
                self.config.noise_reduction_enabled
            ),
            "noise_reduction_strength": (
                self.config.noise_reduction_strength
            ),
            "low_cut_hz": (
                self.config.low_cut_hz
            ),
            "high_cut_hz": (
                self.config.high_cut_hz
            ),
        }

    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    def shutdown(
        self,
    ) -> None:

        self.reset_stream()

        self.clear_noise_profile()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def clean_audio(
    samples: Any,
    *,
    sample_rate: int = 16000,
    config: Optional[
        NoiseCancellationConfig
    ] = None,
) -> Any:
    """
    Convenience function.

    Returns only the cleaned samples.
    """

    processor = NoiseCancellation(
        config
        or NoiseCancellationConfig(
            sample_rate=sample_rate
        )
    )

    try:

        result = processor.process(
            samples,
            sample_rate=sample_rate,
        )

        return result.samples

    finally:

        processor.shutdown()


# ============================================================================
# ALIASES
# ============================================================================


NoiseCanceller = NoiseCancellation


__all__ = [
    "NoiseCancellationConfig",
    "NoiseProfile",
    "NoiseCancellationResult",
    "NoiseCancellationError",
    "NoiseCancellation",
    "NoiseCanceller",
    "clean_audio",
    "AudioMath",
]


