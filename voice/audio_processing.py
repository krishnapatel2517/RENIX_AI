"""
RENIX AI - Audio Processing

Central audio-processing utilities used by the RENIX voice pipeline.

Responsibilities:
- Convert microphone/audio input into a consistent format
- Normalize sample data
- Convert integer PCM formats to floating-point audio
- Convert floating-point audio back to PCM
- Resample audio
- Convert stereo/multichannel audio to mono
- Calculate audio metrics
- Split audio into frames
- Apply gain
- Remove DC offset
- Detect clipping
- Detect silence
- Apply simple fade-in/fade-out
- Mix audio buffers
- Prepare audio for STT, speaker identification and authentication
- Provide streaming-safe processing state

Pipeline:

    Microphone
        ↓
    audio_processing.py
        ↓
    noise_cancellation.py
        ↓
    speech_to_text.py
    speaker_identification.py
    voice_authentication.py
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Literal, Optional, Sequence

logger = logging.getLogger(
    "RENIX.voice.audio_processing"
)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class AudioProcessingConfig:
    """Configuration for RENIX audio processing."""

    input_sample_rate: int = 16000

    output_sample_rate: int = 16000

    channels: int = 1

    target_channels: int = 1

    sample_width: int = 2

    normalize: bool = True

    remove_dc: bool = True

    target_peak: float = 0.95

    silence_threshold: float = 0.01

    frame_duration_ms: float = 20.0

    max_amplitude: float = 1.0

    clipping_threshold: float = 0.99


# ============================================================================
# DATA TYPES
# ============================================================================


@dataclass
class AudioMetrics:
    """Basic measurements for an audio buffer."""

    sample_count: int

    duration_seconds: float

    rms: float

    peak: float

    mean: float

    variance: float

    zero_crossing_rate: float

    is_silent: bool

    is_clipped: bool

    clipping_ratio: float

    sample_rate: int

    channels: int

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "sample_count": self.sample_count,
            "duration_seconds": self.duration_seconds,
            "rms": self.rms,
            "peak": self.peak,
            "mean": self.mean,
            "variance": self.variance,
            "zero_crossing_rate": (
                self.zero_crossing_rate
            ),
            "is_silent": self.is_silent,
            "is_clipped": self.is_clipped,
            "clipping_ratio": (
                self.clipping_ratio
            ),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "metadata": self.metadata,
        }


@dataclass
class AudioFrame:
    """One frame of processed audio."""

    samples: list[float]

    index: int

    start_time: float

    end_time: float

    sample_rate: int

    channels: int

    is_silent: bool = False

    rms: float = 0.0

    peak: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def duration(
        self,
    ) -> float:

        return max(
            0.0,
            self.end_time
            - self.start_time,
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "index": self.index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "is_silent": self.is_silent,
            "rms": self.rms,
            "peak": self.peak,
            "sample_count": len(
                self.samples
            ),
            "metadata": self.metadata,
        }


@dataclass
class AudioBuffer:
    """Standardized RENIX audio buffer."""

    samples: list[float]

    sample_rate: int

    channels: int = 1

    normalized: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def sample_count(
        self,
    ) -> int:

        return len(
            self.samples
        )

    @property
    def duration(
        self,
    ) -> float:

        if self.sample_rate <= 0:

            return 0.0

        return (
            len(self.samples)
            / self.sample_rate
            / max(
                1,
                self.channels,
            )
        )

    def copy(
        self,
    ) -> "AudioBuffer":

        return AudioBuffer(
            samples=self.samples.copy(),
            sample_rate=self.sample_rate,
            channels=self.channels,
            normalized=self.normalized,
            metadata=dict(
                self.metadata
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "sample_count": self.sample_count,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration": self.duration,
            "normalized": self.normalized,
            "metadata": self.metadata,
        }


class AudioProcessingError(
    RuntimeError
):
    """Base audio-processing exception."""


# ============================================================================
# AUDIO PROCESSING ENGINE
# ============================================================================


class AudioProcessor:
    """
    Main RENIX audio-processing engine.
    """

    def __init__(
        self,
        config: Optional[
            AudioProcessingConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or AudioProcessingConfig()
        )

        self._lock = threading.RLock()

        self._stream_remainder: list[
            float
        ] = []

        self._stream_frame_index = 0

        self._last_buffer: Optional[
            AudioBuffer
        ] = None

        self._validate_config()

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def _validate_config(
        self,
    ) -> None:

        if (
            self.config.input_sample_rate
            <= 0
        ):

            raise ValueError(
                "input_sample_rate must be positive."
            )

        if (
            self.config.output_sample_rate
            <= 0
        ):

            raise ValueError(
                "output_sample_rate must be positive."
            )

        if self.config.channels <= 0:

            raise ValueError(
                "channels must be positive."
            )

        if self.config.target_channels <= 0:

            raise ValueError(
                "target_channels must be positive."
            )

        if not (
            0.0
            < self.config.target_peak
            <= 1.0
        ):

            raise ValueError(
                "target_peak must be between 0 and 1."
            )

        if not (
            0.0
            <= self.config.silence_threshold
            <= 1.0
        ):

            raise ValueError(
                "silence_threshold must be between 0 and 1."
            )

        if self.config.frame_duration_ms <= 0:

            raise ValueError(
                "frame_duration_ms must be positive."
            )

        if not (
            0.0
            < self.config.max_amplitude
            <= 1.0
        ):

            raise ValueError(
                "max_amplitude must be between 0 and 1."
            )

    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================

    def process(
        self,
        samples: Any,
        *,
        input_sample_rate: Optional[
            int
        ] = None,
        input_channels: Optional[
            int
        ] = None,
        sample_width: Optional[
            int
        ] = None,
    ) -> AudioBuffer:

        rate = int(
            input_sample_rate
            or self.config.input_sample_rate
        )

        channels = int(
            input_channels
            or self.config.channels
        )

        width = int(
            sample_width
            or self.config.sample_width
        )

        values = self.to_float(
            samples,
            sample_width=width,
        )

        self._validate_samples(
            values
        )

        if channels > 1:

            values = self.to_mono(
                values,
                channels,
            )

            channels = 1

        if self.config.remove_dc:

            values = self.remove_dc(
                values
            )

        if (
            rate
            != self.config.output_sample_rate
        ):

            values = self.resample(
                values,
                rate,
                self.config.output_sample_rate,
            )

            rate = (
                self.config.output_sample_rate
            )

        if self.config.normalize:

            values = self.normalize(
                values,
                self.config.target_peak,
            )

        values = [
            self.clamp(
                value,
                -self.config.max_amplitude,
                self.config.max_amplitude,
            )
            for value in values
        ]

        buffer = AudioBuffer(
            samples=values,
            sample_rate=rate,
            channels=channels,
            normalized=True,
            metadata={
                "source_sample_rate": (
                    input_sample_rate
                    or self.config.input_sample_rate
                ),
                "source_channels": (
                    input_channels
                    or self.config.channels
                ),
                "sample_width": width,
            },
        )

        self._last_buffer = buffer

        return buffer

    @staticmethod
    def _validate_samples(
        samples: Iterable[float],
    ) -> None:
        """Validate normalized audio samples before processing."""

        for sample in samples:
            if not math.isfinite(float(sample)):
                raise AudioProcessingError(
                    "Audio samples must be finite numbers."
                )

    # =========================================================================
    # TYPE CONVERSION
    # =========================================================================

    @staticmethod
    def flatten(
        samples: Any,
    ) -> list[float]:

        if samples is None:

            return []

        if isinstance(
            samples,
            (bytes, bytearray),
        ):

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

                    values.extend(
                        float(value)
                        for value in item
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
    def to_float(
        samples: Any,
        *,
        sample_width: int = 2,
    ) -> list[float]:

        if samples is None:

            return []

        if isinstance(
            samples,
            (bytes, bytearray),
        ):

            return AudioProcessor.pcm_bytes_to_float(
                bytes(samples),
                sample_width=sample_width,
            )

        values = AudioProcessor.flatten(
            samples
        )

        if not values:

            return []

        # If values are already in the normal [-1, 1] range,
        # keep them as floating-point samples.
        peak = max(
            abs(value)
            for value in values
        )

        if peak <= 1.0:

            return values

        # Otherwise interpret them as integer PCM.
        maximum = (
            float(
                2 ** (
                    sample_width * 8
                    - 1
                )
                - 1
            )
        )

        if maximum <= 0:

            raise AudioProcessingError(
                "Invalid sample width."
            )

        return [
            value / maximum
            for value in values
        ]

    @staticmethod
    def pcm_bytes_to_float(
        data: bytes,
        *,
        sample_width: int = 2,
        byteorder: Literal["little", "big"] = "little",
        signed: bool = True,
    ) -> list[float]:

        if not data:

            return []

        if sample_width not in (
            1,
            2,
            3,
            4,
        ):

            raise ValueError(
                "sample_width must be 1, 2, 3 or 4 bytes."
            )

        if len(data) % sample_width != 0:

            raise AudioProcessingError(
                "PCM byte buffer length is not "
                "aligned to sample width."
            )

        output = []

        maximum = (
            2 ** (
                sample_width * 8 - 1
            )
            - 1
        )

        minimum = -(
            2 ** (
                sample_width * 8 - 1
            )
        )

        for index in range(
            0,
            len(data),
            sample_width,
        ):

            chunk = data[
                index:
                index + sample_width
            ]

            integer = int.from_bytes(
                chunk,
                byteorder=byteorder,
                signed=signed,
            )

            if not signed:

                integer -= (
                    2 ** (
                        sample_width * 8 - 1
                    )
                )

            integer = max(
                minimum,
                min(
                    maximum,
                    integer,
                ),
            )

            output.append(
                integer
                / maximum
            )

        return output

    @staticmethod
    def float_to_pcm_bytes(
        samples: Iterable[float],
        *,
        sample_width: int = 2,
        byteorder: Literal["little", "big"] = "little",
    ) -> bytes:

        if sample_width not in (
            1,
            2,
            3,
            4,
        ):

            raise ValueError(
                "sample_width must be 1, 2, 3 or 4 bytes."
            )

        maximum = (
            2 ** (
                sample_width * 8 - 1
            )
            - 1
        )

        minimum = -(
            2 ** (
                sample_width * 8 - 1
            )
        )

        output = bytearray()

        for value in samples:

            value = max(
                -1.0,
                min(
                    1.0,
                    float(value),
                ),
            )

            integer = int(
                round(
                    value
                    * maximum
                )
            )

            integer = max(
                minimum,
                min(
                    maximum,
                    integer,
                ),
            )

            output.extend(
                int(integer).to_bytes(
                    sample_width,
                    byteorder=byteorder,
                    signed=True,
                )
            )

        return bytes(
            output
        )

    # =========================================================================
    # CHANNEL PROCESSING
    # =========================================================================

    @staticmethod
    def to_mono(
        samples: Any,
        channels: int,
    ) -> list[float]:

        values = AudioProcessor.flatten(
            samples
        )

        if channels <= 0:

            raise ValueError(
                "channels must be positive."
            )

        if channels == 1:

            return values

        if not values:

            return []

        frame_count = (
            len(values)
            // channels
        )

        output = []

        for frame_index in range(
            frame_count
        ):

            start = (
                frame_index
                * channels
            )

            frame = values[
                start:
                start + channels
            ]

            if not frame:

                continue

            output.append(
                sum(frame)
                / len(frame)
            )

        return output

    @staticmethod
    def mono_to_channels(
        samples: Any,
        channels: int,
    ) -> list[float]:

        values = AudioProcessor.flatten(
            samples
        )

        if channels <= 0:

            raise ValueError(
                "channels must be positive."
            )

        if channels == 1:

            return values

        output = []

        for value in values:

            for _ in range(
                channels
            ):

                output.append(
                    value
                )

        return output

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    @staticmethod
    def normalize(
        samples: Any,
        target_peak: float = 0.95,
    ) -> list[float]:

        values = AudioProcessor.flatten(
            samples
        )

        if not values:

            return []

        if not (
            0.0
            < target_peak
            <= 1.0
        ):

            raise ValueError(
                "target_peak must be between 0 and 1."
            )

        peak = max(
            abs(value)
            for value in values
        )

        if peak <= 1e-12:

            return values

        gain = (
            target_peak
            / peak
        )

        return [
            AudioProcessor.clamp(
                value * gain
            )
            for value in values
        ]

    @staticmethod
    def gain(
        samples: Any,
        gain_value: float,
    ) -> list[float]:

        values = AudioProcessor.flatten(
            samples
        )

        return [
            AudioProcessor.clamp(
                value * gain_value
            )
            for value in values
        ]

    # =========================================================================
    # DC OFFSET
    # =========================================================================

    @staticmethod
    def remove_dc(
        samples: Any,
    ) -> list[float]:

        values = AudioProcessor.flatten(
            samples
        )

        if not values:

            return []

        mean = (
            sum(values)
            / len(values)
        )

        return [
            value - mean
            for value in values
        ]

    # =========================================================================
    # RESAMPLING
    # =========================================================================

    @staticmethod
    def resample(
        samples: Any,
        input_rate: int,
        output_rate: int,
    ) -> list[float]:

        values = AudioProcessor.flatten(
            samples
        )

        if not values:

            return []

        if input_rate <= 0:

            raise ValueError(
                "input_rate must be positive."
            )

        if output_rate <= 0:

            raise ValueError(
                "output_rate must be positive."
            )

        if input_rate == output_rate:

            return values.copy()

        if len(values) == 1:

            return values.copy()

        output_length = max(
            1,
            int(
                round(
                    len(values)
                    * output_rate
                    / input_rate
                )
            ),
        )

        output = []

        ratio = (
            input_rate
            / output_rate
        )

        last_index = (
            len(values) - 1
        )

        for index in range(
            output_length
        ):

            source_position = (
                index
                * ratio
            )

            left = int(
                math.floor(
                    source_position
                )
            )

            right = min(
                last_index,
                left + 1,
            )

            fraction = (
                source_position
                - left
            )

            if left > last_index:

                value = values[
                    last_index
                ]

            else:

                value = (
                    values[left]
                    * (1.0 - fraction)
                    + values[right]
                    * fraction
                )

            output.append(
                value
            )

        return output

    # =========================================================================
    # METRICS
    # =========================================================================

    @staticmethod
    def rms(
        samples: Any,
    ) -> float:

        values = AudioProcessor.flatten(
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
        samples: Any,
    ) -> float:

        values = AudioProcessor.flatten(
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
        samples: Any,
    ) -> float:

        values = AudioProcessor.flatten(
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
        samples: Any,
    ) -> float:

        values = AudioProcessor.flatten(
            samples
        )

        if len(values) < 2:

            return 0.0

        mean = (
            sum(values)
            / len(values)
        )

        return (
            sum(
                (
                    value
                    - mean
                ) ** 2
                for value in values
            )
            / len(values)
        )

    @staticmethod
    def zero_crossing_rate(
        samples: Any,
    ) -> float:

        values = AudioProcessor.flatten(
            samples
        )

        if len(values) < 2:

            return 0.0

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
            / (
                len(values) - 1
            )
        )

    def metrics(
        self,
        samples: Any,
        *,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
    ) -> AudioMetrics:

        values = AudioProcessor.flatten(
            samples
        )

        rate = int(
            sample_rate
            or self.config.output_sample_rate
        )

        channel_count = int(
            channels
            or self.config.target_channels
        )

        rms = self.rms(
            values
        )

        peak = self.peak(
            values
        )

        clipping_count = sum(
            1
            for value in values
            if abs(value)
            >= self.config.clipping_threshold
        )

        clipping_ratio = (
            clipping_count
            / len(values)
            if values
            else 0.0
        )

        duration = (
            len(values)
            / rate
            / max(
                1,
                channel_count,
            )
        )

        return AudioMetrics(
            sample_count=len(
                values
            ),
            duration_seconds=duration,
            rms=rms,
            peak=peak,
            mean=self.mean(
                values
            ),
            variance=self.variance(
                values
            ),
            zero_crossing_rate=(
                self.zero_crossing_rate(
                    values
                )
            ),
            is_silent=(
                rms
                <= self.config.silence_threshold
            ),
            is_clipped=(
                clipping_count > 0
            ),
            clipping_ratio=clipping_ratio,
            sample_rate=rate,
            channels=channel_count,
        )

    # =========================================================================
    # SILENCE / CLIPPING
    # =========================================================================

    def is_silent(
        self,
        samples: Any,
        *,
        threshold: Optional[
            float
        ] = None,
    ) -> bool:

        value = (
            threshold
            if threshold is not None
            else self.config.silence_threshold
        )

        return (
            self.rms(samples)
            <= value
        )

    def is_clipped(
        self,
        samples: Any,
        *,
        threshold: Optional[
            float
        ] = None,
    ) -> bool:

        values = AudioProcessor.flatten(
            samples
        )

        if not values:

            return False

        limit = (
            threshold
            if threshold is not None
            else self.config.clipping_threshold
        )

        return any(
            abs(value)
            >= limit
            for value in values
        )

    def clipping_ratio(
        self,
        samples: Any,
        *,
        threshold: Optional[
            float
        ] = None,
    ) -> float:

        values = AudioProcessor.flatten(
            samples
        )

        if not values:

            return 0.0

        limit = (
            threshold
            if threshold is not None
            else self.config.clipping_threshold
        )

        clipped = sum(
            1
            for value in values
            if abs(value)
            >= limit
        )

        return (
            clipped
            / len(values)
        )

    # =========================================================================
    # FRAMING
    # =========================================================================

    def frame_size(
        self,
        sample_rate: Optional[int] = None,
        duration_ms: Optional[float] = None,
    ) -> int:

        rate = int(
            sample_rate
            or self.config.output_sample_rate
        )

        duration = (
            duration_ms
            if duration_ms is not None
            else self.config.frame_duration_ms
        )

        return max(
            1,
            int(
                round(
                    rate
                    * duration
                    / 1000.0
                )
            ),
        )

    def frame_audio(
        self,
        samples: Any,
        *,
        sample_rate: Optional[int] = None,
        frame_duration_ms: Optional[
            float
        ] = None,
        include_partial: bool = True,
    ) -> list[AudioFrame]:

        values = AudioProcessor.flatten(
            samples
        )

        rate = int(
            sample_rate
            or self.config.output_sample_rate
        )

        size = self.frame_size(
            rate,
            frame_duration_ms,
        )

        if not values:

            return []

        frames = []

        index = 0

        start = 0

        while start < len(values):

            end = min(
                start + size,
                len(values),
            )

            chunk = values[
                start:end
            ]

            if (
                len(chunk) < size
                and not include_partial
            ):

                break

            rms = self.rms(
                chunk
            )

            peak = self.peak(
                chunk
            )

            start_time = (
                start / rate
            )

            end_time = (
                end / rate
            )

            frames.append(
                AudioFrame(
                    samples=chunk,
                    index=index,
                    start_time=start_time,
                    end_time=end_time,
                    sample_rate=rate,
                    channels=1,
                    is_silent=(
                        rms
                        <= self.config.silence_threshold
                    ),
                    rms=rms,
                    peak=peak,
                )
            )

            start = end

            index += 1

        return frames

    # =========================================================================
    # STREAMING FRAMING
    # =========================================================================

    def stream_frames(
        self,
        samples: Any,
        *,
        sample_rate: Optional[int] = None,
        frame_duration_ms: Optional[
            float
        ] = None,
    ) -> Iterator[AudioFrame]:

        values = AudioProcessor.flatten(
            samples
        )

        rate = int(
            sample_rate
            or self.config.output_sample_rate
        )

        size = self.frame_size(
            rate,
            frame_duration_ms,
        )

        with self._lock:

            self._stream_remainder.extend(
                values
            )

            while (
                len(
                    self._stream_remainder
                )
                >= size
            ):

                chunk = (
                    self._stream_remainder[
                        :size
                    ]
                )

                del self._stream_remainder[
                    :size
                ]

                index = (
                    self._stream_frame_index
                )

                self._stream_frame_index += 1

                start_time = (
                    index
                    * size
                    / rate
                )

                end_time = (
                    start_time
                    + size
                    / rate
                )

                rms = self.rms(
                    chunk
                )

                yield AudioFrame(
                    samples=chunk,
                    index=index,
                    start_time=start_time,
                    end_time=end_time,
                    sample_rate=rate,
                    channels=1,
                    is_silent=(
                        rms
                        <= self.config.silence_threshold
                    ),
                    rms=rms,
                    peak=self.peak(
                        chunk
                    ),
                )

    def flush_stream(
        self,
        *,
        sample_rate: Optional[int] = None,
    ) -> Optional[AudioFrame]:

        with self._lock:

            if not self._stream_remainder:

                return None

            rate = int(
                sample_rate
                or self.config.output_sample_rate
            )

            chunk = (
                self._stream_remainder.copy()
            )

            self._stream_remainder.clear()

            index = (
                self._stream_frame_index
            )

            self._stream_frame_index += 1

        start_time = (
            index
            * self.frame_size(
                rate
            )
            / rate
        )

        end_time = (
            start_time
            + len(chunk)
            / rate
        )

        return AudioFrame(
            samples=chunk,
            index=index,
            start_time=start_time,
            end_time=end_time,
            sample_rate=rate,
            channels=1,
            is_silent=(
                self.is_silent(
                    chunk
                )
            ),
            rms=self.rms(
                chunk
            ),
            peak=self.peak(
                chunk
            ),
        )

    # =========================================================================
    # FADES
    # =========================================================================

    @staticmethod
    def fade_in(
        samples: Any,
        *,
        duration_samples: int,
    ) -> list[float]:

        values = AudioProcessor.flatten(
            samples
        )

        if not values:

            return []

        duration_samples = max(
            0,
            int(
                duration_samples
            ),
        )

        if duration_samples <= 1:

            return values

        duration_samples = min(
            duration_samples,
            len(values),
        )

        output = values.copy()

        for index in range(
            duration_samples
        ):

            factor = (
                index
                / (
                    duration_samples
                    - 1
                )
                if duration_samples > 1
                else 1.0
            )

            output[index] *= factor

        return output

    @staticmethod
    def fade_out(
        samples: Any,
        *,
        duration_samples: int,
    ) -> list[float]:

        values = AudioProcessor.flatten(
            samples
        )

        if not values:

            return []

        duration_samples = max(
            0,
            int(
                duration_samples
            ),
        )

        if duration_samples <= 1:

            return values

        duration_samples = min(
            duration_samples,
            len(values),
        )

        output = values.copy()

        start = (
            len(values)
            - duration_samples
        )

        for offset in range(
            duration_samples
        ):

            factor = (
                1.0
                - (
                    offset
                    / (
                        duration_samples
                        - 1
                    )
                )
            )

            output[
                start + offset
            ] *= factor

        return output

    # =========================================================================
    # MIXING
    # =========================================================================

    @staticmethod
    def mix(
        *buffers: Any,
        gains: Optional[
            Sequence[float]
        ] = None,
    ) -> list[float]:

        if not buffers:

            return []

        normalized = [
            AudioProcessor.flatten(
                buffer
            )
            for buffer in buffers
        ]

        normalized = [
            buffer
            for buffer in normalized
            if buffer
        ]

        if not normalized:

            return []

        length = max(
            len(buffer)
            for buffer in normalized
        )

        if gains is None:

            gains = [
                1.0
                for _ in normalized
            ]

        if len(gains) != len(
            normalized
        ):

            raise ValueError(
                "gains length must match "
                "number of buffers."
            )

        output = [
            0.0
            for _ in range(
                length
            )
        ]

        for buffer, gain_value in zip(
            normalized,
            gains,
        ):

            for index, value in enumerate(
                buffer
            ):

                output[index] += (
                    value
                    * gain_value
                )

        return [
            AudioProcessor.clamp(
                value
            )
            for value in output
        ]

    # =========================================================================
    # SILENCE GENERATION
    # =========================================================================

    @staticmethod
    def silence(
        duration_seconds: float,
        sample_rate: int = 16000,
    ) -> list[float]:

        if duration_seconds < 0:

            raise ValueError(
                "duration_seconds cannot be negative."
            )

        if sample_rate <= 0:

            raise ValueError(
                "sample_rate must be positive."
            )

        count = int(
            round(
                duration_seconds
                * sample_rate
            )
        )

        return [
            0.0
            for _ in range(
                count
            )
        ]

    # =========================================================================
    # CLAMP
    # =========================================================================

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

    # =========================================================================
    # STREAM CONTROL
    # =========================================================================

    def reset_stream(
        self,
    ) -> None:

        with self._lock:

            self._stream_remainder.clear()

            self._stream_frame_index = 0

    def get_stream_buffer_size(
        self,
    ) -> int:

        with self._lock:

            return len(
                self._stream_remainder
            )

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_last_buffer(
        self,
    ) -> Optional[AudioBuffer]:

        return self._last_buffer

    def get_status(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "input_sample_rate": (
                    self.config.input_sample_rate
                ),
                "output_sample_rate": (
                    self.config.output_sample_rate
                ),
                "channels": (
                    self.config.channels
                ),
                "target_channels": (
                    self.config.target_channels
                ),
                "frame_duration_ms": (
                    self.config.frame_duration_ms
                ),
                "frame_size": (
                    self.frame_size()
                ),
                "stream_buffer_size": (
                    len(
                        self._stream_remainder
                    )
                ),
                "stream_frame_index": (
                    self._stream_frame_index
                ),
                "last_buffer_available": (
                    self._last_buffer
                    is not None
                ),
            }

    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    def shutdown(
        self,
    ) -> None:

        self.reset_stream()

        with self._lock:

            self._last_buffer = None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def process_audio(
    samples: Any,
    *,
    input_sample_rate: int = 16000,
    output_sample_rate: int = 16000,
    channels: int = 1,
) -> AudioBuffer:
    """
    Process an audio buffer using the default RENIX pipeline.
    """

    processor = AudioProcessor(
        AudioProcessingConfig(
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            channels=channels,
            target_channels=1,
        )
    )

    try:

        return processor.process(
            samples,
            input_sample_rate=input_sample_rate,
            input_channels=channels,
        )

    finally:

        processor.shutdown()


def calculate_audio_metrics(
    samples: Any,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
) -> AudioMetrics:

    processor = AudioProcessor(
        AudioProcessingConfig(
            output_sample_rate=sample_rate,
            target_channels=channels,
        )
    )

    try:

        return processor.metrics(
            samples,
            sample_rate=sample_rate,
            channels=channels,
        )

    finally:

        processor.shutdown()


def pcm_to_float(
    data: bytes,
    *,
    sample_width: int = 2,
) -> list[float]:

    return AudioProcessor.pcm_bytes_to_float(
        data,
        sample_width=sample_width,
    )


def float_to_pcm(
    samples: Iterable[float],
    *,
    sample_width: int = 2,
) -> bytes:

    return AudioProcessor.float_to_pcm_bytes(
        samples,
        sample_width=sample_width,
    )


# ============================================================================
# ALIASES
# ============================================================================


AudioProcessing = AudioProcessor


__all__ = [
    "AudioProcessingConfig",
    "AudioMetrics",
    "AudioFrame",
    "AudioBuffer",
    "AudioProcessingError",
    "AudioProcessor",
    "AudioProcessing",
    "process_audio",
    "calculate_audio_metrics",
    "pcm_to_float",
    "float_to_pcm",
]


