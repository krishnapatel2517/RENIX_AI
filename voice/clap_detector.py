"""
RENIX AI - Clap Detector

Detects intentional hand claps from microphone audio.

Primary RENIX use case:
    TWO CLAPS -> wake/activate RENIX

Features:
- Audio-stream based clap detection
- Double-clap detection
- Configurable amplitude threshold
- Configurable frequency range
- Minimum/maximum interval between claps
- Cooldown/debouncing
- Optional microphone callback
- Background listening
- Thread-safe state management
- Works with sounddevice when installed
- Graceful fallback when audio dependencies are unavailable

This module is intentionally independent from wake_word.py.

Architecture:

    Microphone
        |
        v
    Audio samples
        |
        v
    ClapDetector
        |
        +---- single clap
        |
        +---- double clap
                  |
                  v
             RENIX activation
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("RENIX.voice.clap_detector")


# ============================================================================
# DATA TYPES
# ============================================================================


@dataclass
class ClapDetectorConfig:
    """Configuration for clap detection."""

    sample_rate: int = 16000

    channels: int = 1

    block_size: int = 1024

    # Minimum RMS amplitude considered a possible clap.
    amplitude_threshold: float = 0.18

    # Peak amplitude required for a strong transient.
    peak_threshold: float = 0.45

    # Minimum distance between two individual clap detections.
    min_clap_interval: float = 0.08

    # Maximum distance between claps to count as a double clap.
    max_double_clap_interval: float = 0.75

    # Number of claps required for activation.
    required_claps: int = 2

    # Prevent repeated double-clap activations.
    cooldown: float = 1.5

    # How long a detected clap remains in the rolling history.
    history_window: float = 1.5

    # Number of consecutive high-energy blocks needed.
    min_transient_blocks: int = 1

    # Optional high-frequency preference.
    use_frequency_filter: bool = False

    low_frequency: float = 1200.0

    high_frequency: float = 8000.0

    # Background listener settings.
    polling_interval: float = 0.01

    # Audio device. None means default microphone.
    device: Optional[Any] = None

    enabled: bool = True


@dataclass
class ClapEvent:
    """Information about a detected clap."""

    timestamp: float

    amplitude: float

    peak: float

    confidence: float

    clap_number: int

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "amplitude": self.amplitude,
            "peak": self.peak,
            "confidence": self.confidence,
            "clap_number": self.clap_number,
            "metadata": self.metadata,
        }


@dataclass
class DoubleClapEvent:
    """Information about a detected double clap."""

    first_clap: ClapEvent

    second_clap: ClapEvent

    interval: float

    confidence: float

    timestamp: float = field(
        default_factory=time.time
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_clap": self.first_clap.to_dict(),
            "second_clap": self.second_clap.to_dict(),
            "interval": self.interval,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


class ClapDetectorError(RuntimeError):
    """Base clap detector error."""


class AudioInputUnavailableError(
    ClapDetectorError
):
    """Raised when microphone/audio input is unavailable."""


# ============================================================================
# MAIN CLASS
# ============================================================================


class ClapDetector:
    """
    RENIX clap detection engine.

    Example:

        detector = ClapDetector()

        detector.add_double_clap_callback(
            lambda event: print("RENIX WAKE")
        )

        detector.start()

    A manual audio block can also be supplied:

        detector.process_audio(samples)
    """

    def __init__(
        self,
        config: Optional[ClapDetectorConfig] = None,
    ) -> None:

        self.config = (
            config
            or ClapDetectorConfig()
        )

        self._lock = threading.RLock()

        self._running = False

        self._stop_event = threading.Event()

        self._worker_thread: Optional[
            threading.Thread
        ] = None

        self._sounddevice = None

        self._audio_stream = None

        self._clap_history: deque[
            ClapEvent
        ] = deque()

        self._clap_count = 0

        self._last_clap_time = 0.0

        self._last_double_clap_time = 0.0

        self._last_clap: Optional[
            ClapEvent
        ] = None

        self._last_double_clap: Optional[
            DoubleClapEvent
        ] = None

        self._single_clap_callbacks: list[
            Callable[[ClapEvent], None]
        ] = []

        self._double_clap_callbacks: list[
            Callable[[DoubleClapEvent], None]
        ] = []

        self._audio_error: Optional[str] = None

        self._load_audio_backend()

        self._validate_config()

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def _load_audio_backend(self) -> None:

        try:

            import sounddevice

            self._sounddevice = sounddevice

        except ImportError:

            logger.debug(
                "sounddevice is not installed."
            )

            self._sounddevice = None

    def _validate_config(self) -> None:

        if self.config.sample_rate <= 0:

            raise ValueError(
                "sample_rate must be positive."
            )

        if self.config.channels <= 0:

            raise ValueError(
                "channels must be positive."
            )

        if self.config.block_size <= 0:

            raise ValueError(
                "block_size must be positive."
            )

        if self.config.amplitude_threshold < 0:

            raise ValueError(
                "amplitude_threshold cannot be negative."
            )

        if self.config.peak_threshold < 0:

            raise ValueError(
                "peak_threshold cannot be negative."
            )

        if (
            self.config.min_clap_interval
            < 0
        ):

            raise ValueError(
                "min_clap_interval cannot be negative."
            )

        if (
            self.config.max_double_clap_interval
            <= 0
        ):

            raise ValueError(
                "max_double_clap_interval must be positive."
            )

        if self.config.required_claps <= 0:

            raise ValueError(
                "required_claps must be positive."
            )

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def configure(
        self,
        *,
        amplitude_threshold: Optional[
            float
        ] = None,
        peak_threshold: Optional[
            float
        ] = None,
        min_clap_interval: Optional[
            float
        ] = None,
        max_double_clap_interval: Optional[
            float
        ] = None,
        cooldown: Optional[float] = None,
        required_claps: Optional[int] = None,
        sample_rate: Optional[int] = None,
        block_size: Optional[int] = None,
    ) -> ClapDetectorConfig:

        if amplitude_threshold is not None:

            self.config.amplitude_threshold = (
                float(amplitude_threshold)
            )

        if peak_threshold is not None:

            self.config.peak_threshold = (
                float(peak_threshold)
            )

        if min_clap_interval is not None:

            self.config.min_clap_interval = (
                float(min_clap_interval)
            )

        if max_double_clap_interval is not None:

            self.config.max_double_clap_interval = (
                float(max_double_clap_interval)
            )

        if cooldown is not None:

            self.config.cooldown = float(
                cooldown
            )

        if required_claps is not None:

            self.config.required_claps = int(
                required_claps
            )

        if sample_rate is not None:

            self.config.sample_rate = int(
                sample_rate
            )

        if block_size is not None:

            self.config.block_size = int(
                block_size
            )

        self._validate_config()

        return self.config

    # =========================================================================
    # AUDIO ANALYSIS
    # =========================================================================

    def process_audio(
        self,
        samples: Any,
        *,
        timestamp: Optional[float] = None,
    ) -> Optional[ClapEvent]:
        """
        Process one audio block.

        `samples` may be:
        - list
        - tuple
        - NumPy ndarray
        - mono audio
        - multi-channel audio

        Returns a ClapEvent when a clap is detected.
        """

        if not self.config.enabled:

            return None

        if samples is None:

            return None

        try:

            values = self._flatten_audio(
                samples
            )

        except Exception:

            logger.debug(
                "Unable to process audio block.",
                exc_info=True,
            )

            return None

        if not values:

            return None

        amplitude = self._calculate_rms(
            values
        )

        peak = self._calculate_peak(
            values
        )

        # A clap is generally a short, high-energy transient.
        if (
            amplitude
            < self.config.amplitude_threshold
            and peak
            < self.config.peak_threshold
        ):

            return None

        now = (
            timestamp
            if timestamp is not None
            else time.monotonic()
        )

        with self._lock:

            # Debounce.
            if (
                self._last_clap_time
                and (
                    now
                    - self._last_clap_time
                    < self.config.min_clap_interval
                )
            ):

                return None

            self._last_clap_time = now

            self._clap_count += 1

            confidence = (
                self._calculate_confidence(
                    amplitude,
                    peak,
                )
            )

            event = ClapEvent(
                timestamp=now,
                amplitude=amplitude,
                peak=peak,
                confidence=confidence,
                clap_number=self._clap_count,
            )

            self._last_clap = event

            self._clap_history.append(
                event
            )

            self._cleanup_history(
                now
            )

        self._notify_single_clap(
            event
        )

        self._check_double_clap()

        return event

    @staticmethod
    def _flatten_audio(
        samples: Any,
    ) -> list[float]:

        # NumPy arrays.
        if hasattr(
            samples,
            "flatten",
        ):

            flattened = samples.flatten()

            return [
                float(value)
                for value in flattened
            ]

        # Normal Python iterables.
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
    def _calculate_rms(
        samples: list[float],
    ) -> float:

        if not samples:

            return 0.0

        total = 0.0

        for value in samples:

            total += value * value

        return math.sqrt(
            total / len(samples)
        )

    @staticmethod
    def _calculate_peak(
        samples: list[float],
    ) -> float:

        if not samples:

            return 0.0

        return max(
            abs(value)
            for value in samples
        )

    def _calculate_confidence(
        self,
        amplitude: float,
        peak: float,
    ) -> float:

        amplitude_score = 0.0

        peak_score = 0.0

        if self.config.amplitude_threshold > 0:

            amplitude_score = min(
                amplitude
                / self.config.amplitude_threshold,
                2.0,
            )

        if self.config.peak_threshold > 0:

            peak_score = min(
                peak
                / self.config.peak_threshold,
                2.0,
            )

        # Normalize to 0..1.
        score = (
            amplitude_score
            + peak_score
        ) / 4.0

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    # =========================================================================
    # DOUBLE CLAP
    # =========================================================================

    def _check_double_clap(
        self,
    ) -> Optional[DoubleClapEvent]:

        with self._lock:

            if len(
                self._clap_history
            ) < 2:

                return None

            first = self._clap_history[-2]

            second = self._clap_history[-1]

            interval = (
                second.timestamp
                - first.timestamp
            )

            if interval <= 0:

                return None

            if (
                interval
                > self.config.max_double_clap_interval
            ):

                return None

            now = time.monotonic()

            if (
                self._last_double_clap_time
                and (
                    now
                    - self._last_double_clap_time
                    < self.config.cooldown
                )
            ):

                return None

            self._last_double_clap_time = now

            confidence = (
                first.confidence
                + second.confidence
            ) / 2.0

            event = DoubleClapEvent(
                first_clap=first,
                second_clap=second,
                interval=interval,
                confidence=confidence,
            )

            self._last_double_clap = event

        self._notify_double_clap(
            event
        )

        return event

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def add_clap_callback(
        self,
        callback: Callable[
            [ClapEvent],
            None,
        ],
    ) -> None:

        if not callable(callback):

            raise ValueError(
                "callback must be callable."
            )

        with self._lock:

            if callback not in (
                self._single_clap_callbacks
            ):

                self._single_clap_callbacks.append(
                    callback
                )

    def add_double_clap_callback(
        self,
        callback: Callable[
            [DoubleClapEvent],
            None,
        ],
    ) -> None:

        if not callable(callback):

            raise ValueError(
                "callback must be callable."
            )

        with self._lock:

            if callback not in (
                self._double_clap_callbacks
            ):

                self._double_clap_callbacks.append(
                    callback
                )

    def remove_clap_callback(
        self,
        callback: Callable[
            [ClapEvent],
            None,
        ],
    ) -> bool:

        with self._lock:

            if callback in (
                self._single_clap_callbacks
            ):

                self._single_clap_callbacks.remove(
                    callback
                )

                return True

        return False

    def remove_double_clap_callback(
        self,
        callback: Callable[
            [DoubleClapEvent],
            None,
        ],
    ) -> bool:

        with self._lock:

            if callback in (
                self._double_clap_callbacks
            ):

                self._double_clap_callbacks.remove(
                    callback
                )

                return True

        return False

    def clear_callbacks(self) -> None:

        with self._lock:

            self._single_clap_callbacks.clear()

            self._double_clap_callbacks.clear()

    def _notify_single_clap(
        self,
        event: ClapEvent,
    ) -> None:

        with self._lock:

            callbacks = list(
                self._single_clap_callbacks
            )

        for callback in callbacks:

            try:

                callback(event)

            except Exception:

                logger.exception(
                    "Single-clap callback failed."
                )

    def _notify_double_clap(
        self,
        event: DoubleClapEvent,
    ) -> None:

        with self._lock:

            callbacks = list(
                self._double_clap_callbacks
            )

        for callback in callbacks:

            try:

                callback(event)

            except Exception:

                logger.exception(
                    "Double-clap callback failed."
                )

    # =========================================================================
    # MICROPHONE STREAM
    # =========================================================================

    def start(
        self,
        *,
        callback: Optional[
            Callable[[DoubleClapEvent], None]
        ] = None,
    ) -> threading.Thread:
        """
        Start microphone-based clap detection.

        Uses sounddevice.InputStream.
        """

        if self._sounddevice is None:

            raise AudioInputUnavailableError(
                "sounddevice is not installed. "
                "Install it before starting microphone clap detection."
            )

        if callback is not None:

            self.add_double_clap_callback(
                callback
            )

        with self._lock:

            if self._running:

                raise ClapDetectorError(
                    "Clap detector is already running."
                )

            self._running = True

            self._stop_event.clear()

            self._audio_error = None

        try:

            self._audio_stream = (
                self._sounddevice.InputStream(
                    samplerate=self.config.sample_rate,
                    channels=self.config.channels,
                    blocksize=self.config.block_size,
                    device=self.config.device,
                    callback=self._audio_callback,
                )
            )

            self._audio_stream.start()

        except Exception as exc:

            with self._lock:

                self._running = False

                self._audio_error = str(
                    exc
                )

            raise AudioInputUnavailableError(
                str(exc)
            ) from exc

        self._worker_thread = threading.Thread(
            target=self._monitor_stream,
            daemon=True,
            name="RENIX-ClapDetector",
        )

        self._worker_thread.start()

        return self._worker_thread

    def _audio_callback(
        self,
        indata: Any,
        frames: int,
        callback_time: Any,
        status: Any,
    ) -> None:

        if status:

            logger.debug(
                "Audio input status: %s",
                status,
            )

        if self._stop_event.is_set():

            return

        try:

            self.process_audio(
                indata
            )

        except Exception:

            logger.exception(
                "Audio callback processing failed."
            )

    def _monitor_stream(
        self,
    ) -> None:

        while not self._stop_event.is_set():

            self._stop_event.wait(
                self.config.polling_interval
            )

    def stop(self) -> None:

        self._stop_event.set()

        with self._lock:

            self._running = False

        if self._audio_stream is not None:

            try:

                self._audio_stream.stop()

            except Exception:

                logger.debug(
                    "Failed stopping audio stream.",
                    exc_info=True,
                )

            try:

                self._audio_stream.close()

            except Exception:

                logger.debug(
                    "Failed closing audio stream.",
                    exc_info=True,
                )

            self._audio_stream = None

    # =========================================================================
    # HISTORY
    # =========================================================================

    def _cleanup_history(
        self,
        now: float,
    ) -> None:

        while self._clap_history:

            oldest = self._clap_history[0]

            if (
                now
                - oldest.timestamp
                > self.config.history_window
            ):

                self._clap_history.popleft()

            else:

                break

    def clear_history(self) -> None:

        with self._lock:

            self._clap_history.clear()

            self._clap_count = 0

            self._last_clap = None

            self._last_double_clap = None

    def get_clap_history(
        self,
    ) -> list[ClapEvent]:

        with self._lock:

            return list(
                self._clap_history
            )

    # =========================================================================
    # STATE
    # =========================================================================

    def is_running(self) -> bool:

        with self._lock:

            return self._running

    def get_last_clap(
        self,
    ) -> Optional[ClapEvent]:

        return self._last_clap

    def get_last_double_clap(
        self,
    ) -> Optional[DoubleClapEvent]:

        return self._last_double_clap

    def get_status(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "enabled": self.config.enabled,
                "running": self._running,
                "audio_backend_available": (
                    self._sounddevice
                    is not None
                ),
                "sample_rate": (
                    self.config.sample_rate
                ),
                "block_size": (
                    self.config.block_size
                ),
                "amplitude_threshold": (
                    self.config.amplitude_threshold
                ),
                "peak_threshold": (
                    self.config.peak_threshold
                ),
                "min_clap_interval": (
                    self.config.min_clap_interval
                ),
                "max_double_clap_interval": (
                    self.config.max_double_clap_interval
                ),
                "cooldown": (
                    self.config.cooldown
                ),
                "clap_count": (
                    self._clap_count
                ),
                "history_size": (
                    len(self._clap_history)
                ),
                "last_clap": (
                    self._last_clap.to_dict()
                    if self._last_clap
                    else None
                ),
                "last_double_clap": (
                    self._last_double_clap.to_dict()
                    if self._last_double_clap
                    else None
                ),
                "audio_error": (
                    self._audio_error
                ),
            }

    # =========================================================================
    # CALIBRATION
    # =========================================================================

    def calibrate(
        self,
        samples: list[Any],
        *,
        multiplier: float = 1.5,
    ) -> dict[str, float]:
        """
        Estimate thresholds from ambient microphone samples.

        `samples` should contain representative audio blocks captured
        while there is no intentional clap.
        """

        if not samples:

            raise ValueError(
                "Calibration samples cannot be empty."
            )

        rms_values: list[float] = []

        peak_values: list[float] = []

        for block in samples:

            try:

                values = self._flatten_audio(
                    block
                )

                if not values:
                    continue

                rms_values.append(
                    self._calculate_rms(
                        values
                    )
                )

                peak_values.append(
                    self._calculate_peak(
                        values
                    )
                )

            except Exception:

                continue

        if not rms_values:

            raise ValueError(
                "No valid audio samples were provided."
            )

        average_rms = (
            sum(rms_values)
            / len(rms_values)
        )

        average_peak = (
            sum(peak_values)
            / len(peak_values)
        )

        rms_threshold = max(
            0.01,
            average_rms * multiplier,
        )

        peak_threshold = max(
            0.05,
            average_peak * multiplier,
        )

        self.config.amplitude_threshold = (
            rms_threshold
        )

        self.config.peak_threshold = (
            peak_threshold
        )

        return {
            "ambient_rms": average_rms,
            "ambient_peak": average_peak,
            "amplitude_threshold": (
                rms_threshold
            ),
            "peak_threshold": (
                peak_threshold
            ),
        }

    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    def shutdown(self) -> None:

        self.stop()

        self.clear_history()

        self.clear_callbacks()

        with self._lock:

            self._worker_thread = None

            self._audio_error = None

    close = shutdown


# ============================================================================
# FUNCTIONAL HELPERS
# ============================================================================


def detect_clap(
    samples: Any,
    *,
    amplitude_threshold: float = 0.18,
    peak_threshold: float = 0.45,
) -> bool:
    """
    Simple one-shot clap detection helper.
    """

    detector = ClapDetector(
        ClapDetectorConfig(
            amplitude_threshold=(
                amplitude_threshold
            ),
            peak_threshold=(
                peak_threshold
            )
        )
    )

    return (
        detector.process_audio(
            samples
        )
        is not None
    )


def detect_double_clap(
    first_timestamp: float,
    second_timestamp: float,
    *,
    maximum_interval: float = 0.75,
) -> bool:
    """
    Simple timing-based double-clap helper.
    """

    interval = (
        second_timestamp
        - first_timestamp
    )

    return (
        0 < interval <= maximum_interval
    )


# ============================================================================
# ALIASES
# ============================================================================


Clap = ClapEvent

DoubleClap = DoubleClapEvent


__all__ = [
    "ClapDetectorConfig",
    "ClapEvent",
    "DoubleClapEvent",
    "ClapDetectorError",
    "AudioInputUnavailableError",
    "ClapDetector",
    "Clap",
    "DoubleClap",
    "detect_clap",
    "detect_double_clap",
]


