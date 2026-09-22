"""
RENIX AI - Speech To Text

Converts microphone audio into text.

Responsibilities:
- Speech recognition
- Audio-to-text conversion
- Multiple recognition backends
- Language selection
- Confidence handling
- Continuous transcription support
- Graceful fallback when optional packages are unavailable

The module is intentionally backend-agnostic so RENIX can later connect
different STT providers without changing the rest of the voice system.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Sequence

logger = logging.getLogger("RENIX.voice.speech_to_text")


# ============================================================================
# DATA TYPES
# ============================================================================


@dataclass
class STTConfig:
    """Speech-to-text configuration."""

    language: str = "en-US"

    fallback_languages: list[str] = field(
        default_factory=lambda: [
            "en-US",
            "hi-IN",
            "gu-IN",
        ]
    )

    sample_rate: int = 16000

    channels: int = 1

    sample_width: int = 2

    timeout: float = 5.0

    phrase_time_limit: float = 15.0

    energy_threshold: int = 300

    dynamic_energy_threshold: bool = True

    pause_threshold: float = 0.8

    non_speaking_duration: float = 0.5

    ambient_duration: float = 1.0

    preferred_backend: str = "auto"

    enable_google: bool = True

    enable_sphinx: bool = False

    enable_local_fallback: bool = True

    min_text_length: int = 1


@dataclass
class TranscriptionResult:
    """Result returned by RENIX speech recognition."""

    text: str

    success: bool

    language: Optional[str] = None

    confidence: Optional[float] = None

    backend: Optional[str] = None

    duration: Optional[float] = None

    error: Optional[str] = None

    timestamp: float = field(
        default_factory=time.time
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "success": self.success,
            "language": self.language,
            "confidence": self.confidence,
            "backend": self.backend,
            "duration": self.duration,
            "error": self.error,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class SpeechToTextError(RuntimeError):
    """Base STT error."""


class STTUnavailableError(
    SpeechToTextError
):
    """Raised when no STT backend is available."""


class STTRecognitionError(
    SpeechToTextError
):
    """Raised when speech recognition fails."""


class STTTimeoutError(
    SpeechToTextError
):
    """Raised when recognition times out."""


# ============================================================================
# MAIN CLASS
# ============================================================================


class SpeechToText:
    """
    RENIX speech recognition controller.

    Supported input types:

        1. MicrophoneManager
        2. Raw PCM / NumPy audio
        3. WAV bytes
        4. WAV file paths

    Supported recognition backends depend on installed packages.

    The default external backend is Google's speech recognition service
    through the SpeechRecognition package.

    RENIX can later connect a fully local model without changing callers.
    """

    def __init__(
        self,
        config: Optional[STTConfig] = None,
        microphone: Optional[Any] = None,
    ) -> None:

        self.config = (
            config
            or STTConfig()
        )

        self.microphone = microphone

        self._recognizer = None

        self._sr = None

        self._lock = threading.RLock()

        self._running = False

        self._stop_event = threading.Event()

        self._last_result: Optional[
            TranscriptionResult
        ] = None

        self._initialize_speech_recognition()

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def _initialize_speech_recognition(
        self,
    ) -> None:

        try:

            import speech_recognition as sr

            self._sr = sr

            self._recognizer = (
                sr.Recognizer()
            )

            self._recognizer.energy_threshold = (
                self.config.energy_threshold
            )

            self._recognizer.dynamic_energy_threshold = (
                self.config.dynamic_energy_threshold
            )

            self._recognizer.pause_threshold = (
                self.config.pause_threshold
            )

            self._recognizer.non_speaking_duration = (
                self.config.non_speaking_duration
            )

        except ImportError:

            logger.warning(
                "speech_recognition is not installed."
            )

        except Exception:

            logger.exception(
                "Failed initializing speech recognition."
            )

    # =========================================================================
    # AVAILABILITY
    # =========================================================================

    @property
    def available(self) -> bool:
        """Whether a speech recognition backend is available."""

        return self._recognizer is not None

    def is_available(self) -> bool:
        return self.available

    def require_available(self) -> None:

        if not self.available:

            raise STTUnavailableError(
                "No speech-to-text backend is available. "
                "Install SpeechRecognition and a supported backend."
            )

    def get_available_backends(
        self,
    ) -> list[str]:

        backends: list[str] = []

        if self._recognizer is not None:

            if self.config.enable_google:

                backends.append(
                    "google"
                )

            if self.config.enable_sphinx:

                backends.append(
                    "sphinx"
                )

        if self.config.enable_local_fallback:

            backends.append(
                "local"
            )

        return backends

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def set_language(
        self,
        language: str,
    ) -> None:

        language = str(
            language
        ).strip()

        if not language:

            raise ValueError(
                "Language cannot be empty."
            )

        self.config.language = language

    def get_language(self) -> str:

        return self.config.language

    def configure(
        self,
        *,
        language: Optional[str] = None,
        timeout: Optional[float] = None,
        phrase_time_limit: Optional[float] = None,
        energy_threshold: Optional[int] = None,
        pause_threshold: Optional[float] = None,
        backend: Optional[str] = None,
    ) -> STTConfig:

        if language is not None:

            self.set_language(
                language
            )

        if timeout is not None:

            if timeout <= 0:
                raise ValueError(
                    "timeout must be positive."
                )

            self.config.timeout = float(
                timeout
            )

        if phrase_time_limit is not None:

            if phrase_time_limit <= 0:
                raise ValueError(
                    "phrase_time_limit must be positive."
                )

            self.config.phrase_time_limit = (
                float(
                    phrase_time_limit
                )
            )

        if energy_threshold is not None:

            if energy_threshold < 0:
                raise ValueError(
                    "energy_threshold cannot be negative."
                )

            self.config.energy_threshold = int(
                energy_threshold
            )

            if self._recognizer is not None:

                self._recognizer.energy_threshold = (
                    self.config.energy_threshold
                )

        if pause_threshold is not None:

            if pause_threshold <= 0:
                raise ValueError(
                    "pause_threshold must be positive."
                )

            self.config.pause_threshold = (
                float(
                    pause_threshold
                )
            )

            if self._recognizer is not None:

                self._recognizer.pause_threshold = (
                    self.config.pause_threshold
                )

        if backend is not None:

            backend = str(
                backend
            ).lower().strip()

            allowed = {
                "auto",
                "google",
                "sphinx",
                "local",
            }

            if backend not in allowed:

                raise ValueError(
                    "Unsupported STT backend: "
                    f"{backend}"
                )

            self.config.preferred_backend = (
                backend
            )

        return self.config

    # =========================================================================
    # MICROPHONE RECOGNITION
    # =========================================================================

    def listen(
        self,
        *,
        timeout: Optional[float] = None,
        phrase_time_limit: Optional[float] = None,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Listen through the configured microphone and transcribe speech.
        """

        self.require_available()

        timeout = (
            self.config.timeout
            if timeout is None
            else float(timeout)
        )

        phrase_time_limit = (
            self.config.phrase_time_limit
            if phrase_time_limit is None
            else float(phrase_time_limit)
        )

        language = (
            language
            or self.config.language
        )

        if self.microphone is None:

            try:

                with self._sr.Microphone() as source:

                    self._prepare_microphone(
                        source
                    )

                    started = time.monotonic()

                    try:

                        audio = (
                            self._recognizer.listen(
                                source,
                                timeout=timeout,
                                phrase_time_limit=(
                                    phrase_time_limit
                                ),
                            )
                        )

                    except self._sr.WaitTimeoutError as exc:

                        raise STTTimeoutError(
                            "No speech detected before timeout."
                        ) from exc

                    duration = (
                        time.monotonic()
                        - started
                    )

                    return self.recognize_audio(
                        audio,
                        language=language,
                        duration=duration,
                    )

            except STTTimeoutError:

                raise

            except Exception as exc:

                if isinstance(
                    exc,
                    SpeechToTextError,
                ):
                    raise

                raise STTRecognitionError(
                    f"Microphone listening failed: {exc}"
                ) from exc

        return self._listen_from_external_microphone(
            timeout=timeout,
            phrase_time_limit=phrase_time_limit,
            language=language,
        )

    def _prepare_microphone(
        self,
        source: Any,
    ) -> None:

        if self._recognizer is None:
            return

        try:

            self._recognizer.adjust_for_ambient_noise(
                source,
                duration=self.config.ambient_duration,
            )

        except Exception:

            logger.debug(
                "Ambient-noise calibration failed.",
                exc_info=True,
            )

    def _listen_from_external_microphone(
        self,
        *,
        timeout: float,
        phrase_time_limit: float,
        language: str,
    ) -> TranscriptionResult:

        if not hasattr(
            self.microphone,
            "record",
        ):

            raise STTRecognitionError(
                "Configured microphone does not "
                "provide a compatible record() method."
            )

        started = time.monotonic()

        try:

            audio = self.microphone.record(
                phrase_time_limit
            )

        except Exception as exc:

            raise STTRecognitionError(
                f"External microphone recording failed: {exc}"
            ) from exc

        duration = (
            time.monotonic()
            - started
        )

        return self.recognize_raw_audio(
            audio,
            language=language,
            duration=duration,
        )

    # =========================================================================
    # AUDIO RECOGNITION
    # =========================================================================

    def recognize_audio(
        self,
        audio: Any,
        *,
        language: Optional[str] = None,
        duration: Optional[float] = None,
        backend: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Recognize a SpeechRecognition AudioData object.
        """

        self.require_available()

        language = (
            language
            or self.config.language
        )

        backend = (
            backend
            or self.config.preferred_backend
        )

        started = time.monotonic()

        try:

            result = self._recognize_with_backend(
                audio,
                language=language,
                backend=backend,
            )

            if result.duration is None:

                result.duration = (
                    duration
                    if duration is not None
                    else time.monotonic()
                    - started
                )

            self._last_result = result

            return result

        except Exception as exc:

            result = TranscriptionResult(
                text="",
                success=False,
                language=language,
                backend=backend,
                duration=(
                    duration
                    if duration is not None
                    else time.monotonic()
                    - started
                ),
                error=str(exc),
            )

            self._last_result = result

            raise STTRecognitionError(
                str(exc)
            ) from exc

    def recognize_raw_audio(
        self,
        audio: Any,
        *,
        language: Optional[str] = None,
        duration: Optional[float] = None,
        backend: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Convert raw PCM/NumPy audio into SpeechRecognition AudioData.
        """

        self.require_available()

        language = (
            language
            or self.config.language
        )

        try:

            pcm_bytes = self._convert_to_pcm16(
                audio
            )

            audio_data = (
                self._sr.AudioData(
                    pcm_bytes,
                    self.config.sample_rate,
                    self.config.sample_width,
                )
            )

            return self.recognize_audio(
                audio_data,
                language=language,
                duration=duration,
                backend=backend,
            )

        except Exception as exc:

            raise STTRecognitionError(
                f"Raw audio conversion failed: {exc}"
            ) from exc

    # =========================================================================
    # BACKENDS
    # =========================================================================

    def _recognize_with_backend(
        self,
        audio: Any,
        *,
        language: str,
        backend: str,
    ) -> TranscriptionResult:

        backend = (
            str(
                backend
            )
            .lower()
            .strip()
        )

        if backend == "auto":

            return self._recognize_auto(
                audio,
                language=language,
            )

        if backend == "google":

            return self._recognize_google(
                audio,
                language=language,
            )

        if backend == "sphinx":

            return self._recognize_sphinx(
                audio,
                language=language,
            )

        if backend == "local":

            return self._recognize_local(
                audio,
                language=language,
            )

        raise STTRecognitionError(
            f"Unknown STT backend: {backend}"
        )

    def _recognize_auto(
        self,
        audio: Any,
        *,
        language: str,
    ) -> TranscriptionResult:

        errors = []

        if self.config.enable_google:

            try:

                return self._recognize_google(
                    audio,
                    language=language,
                )

            except Exception as exc:

                errors.append(
                    f"google: {exc}"
                )

        if self.config.enable_sphinx:

            try:

                return self._recognize_sphinx(
                    audio,
                    language=language,
                )

            except Exception as exc:

                errors.append(
                    f"sphinx: {exc}"
                )

        if self.config.enable_local_fallback:

            try:

                return self._recognize_local(
                    audio,
                    language=language,
                )

            except Exception as exc:

                errors.append(
                    f"local: {exc}"
                )

        raise STTRecognitionError(
            "All STT backends failed: "
            + " | ".join(errors)
        )

    def _recognize_google(
        self,
        audio: Any,
        *,
        language: str,
    ) -> TranscriptionResult:

        if not self.config.enable_google:

            raise STTRecognitionError(
                "Google recognition is disabled."
            )

        if self._recognizer is None:

            raise STTUnavailableError(
                "SpeechRecognition is unavailable."
            )

        try:

            text = (
                self._recognizer.recognize_google(
                    audio,
                    language=language,
                )
            )

            text = self._clean_text(
                text
            )

            if not text:

                raise STTRecognitionError(
                    "Speech was recognized as empty text."
                )

            return TranscriptionResult(
                text=text,
                success=True,
                language=language,
                confidence=None,
                backend="google",
            )

        except self._sr.UnknownValueError as exc:

            raise STTRecognitionError(
                "Speech could not be understood."
            ) from exc

        except self._sr.RequestError as exc:

            raise STTRecognitionError(
                f"Google STT request failed: {exc}"
            ) from exc

    def _recognize_sphinx(
        self,
        audio: Any,
        *,
        language: str,
    ) -> TranscriptionResult:

        if not self.config.enable_sphinx:

            raise STTRecognitionError(
                "Sphinx recognition is disabled."
            )

        if self._recognizer is None:

            raise STTUnavailableError(
                "SpeechRecognition is unavailable."
            )

        try:

            text = (
                self._recognizer.recognize_sphinx(
                    audio
                )
            )

            text = self._clean_text(
                text
            )

            if not text:

                raise STTRecognitionError(
                    "Sphinx returned empty text."
                )

            return TranscriptionResult(
                text=text,
                success=True,
                language=language,
                confidence=None,
                backend="sphinx",
            )

        except Exception as exc:

            raise STTRecognitionError(
                f"Sphinx STT failed: {exc}"
            ) from exc

    def _recognize_local(
        self,
        audio: Any,
        *,
        language: str,
    ) -> TranscriptionResult:

        """
        Local backend hook.

        A future RENIX local model such as Whisper can be attached here.
        """

        raise STTRecognitionError(
            "No local STT model is configured."
        )

    # =========================================================================
    # WAV SUPPORT
    # =========================================================================

    def recognize_wav_bytes(
        self,
        wav_data: bytes,
        *,
        language: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> TranscriptionResult:

        self.require_available()

        if not wav_data:

            raise ValueError(
                "WAV data cannot be empty."
            )

        try:

            with wave.open(
                io.BytesIO(
                    wav_data
                ),
                "rb",
            ) as wav:

                channels = wav.getnchannels()
                sample_width = (
                    wav.getsampwidth()
                )
                sample_rate = (
                    wav.getframerate()
                )

                frames = wav.readframes(
                    wav.getnframes()
                )

            audio = (
                self._sr.AudioData(
                    frames,
                    sample_rate,
                    sample_width,
                )
            )

            return self.recognize_audio(
                audio,
                language=language,
                backend=backend,
            )

        except Exception as exc:

            raise STTRecognitionError(
                f"WAV recognition failed: {exc}"
            ) from exc

    def recognize_wav_file(
        self,
        path: str,
        *,
        language: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> TranscriptionResult:

        with open(
            path,
            "rb",
        ) as file:

            data = file.read()

        return self.recognize_wav_bytes(
            data,
            language=language,
            backend=backend,
        )

    # =========================================================================
    # RAW AUDIO CONVERSION
    # =========================================================================

    def _convert_to_pcm16(
        self,
        audio: Any,
    ) -> bytes:

        try:

            import numpy as np

            values = np.asarray(
                audio
            )

            if values.size == 0:

                raise ValueError(
                    "Audio is empty."
                )

            values = values.astype(
                np.float32
            )

            if values.ndim > 1:

                values = values.reshape(
                    -1
                )

            values = np.clip(
                values,
                -1.0,
                1.0,
            )

            pcm = (
                values * 32767.0
            ).astype(
                np.int16
            )

            return pcm.tobytes()

        except ImportError as exc:

            raise STTUnavailableError(
                "numpy is required for raw audio conversion."
            ) from exc

    # =========================================================================
    # CONTINUOUS TRANSCRIPTION
    # =========================================================================

    def start_continuous(
        self,
        callback: Callable[
            [TranscriptionResult],
            None,
        ],
        *,
        language: Optional[str] = None,
        interval: float = 0.05,
    ) -> threading.Thread:
        """
        Start continuous microphone transcription.

        `callback` receives every successful or failed transcription result.
        """

        if not callable(callback):

            raise ValueError(
                "callback must be callable."
            )

        with self._lock:

            if self._running:

                raise STTRecognitionError(
                    "Continuous transcription is already running."
                )

            self._running = True

            self._stop_event.clear()

        thread = threading.Thread(
            target=self._continuous_worker,
            args=(
                callback,
                language,
                interval,
            ),
            daemon=True,
            name="RENIX-STT",
        )

        thread.start()

        return thread

    def _continuous_worker(
        self,
        callback: Callable[
            [TranscriptionResult],
            None,
        ],
        language: Optional[str],
        interval: float,
    ) -> None:

        try:

            while not self._stop_event.is_set():

                try:

                    result = self.listen(
                        language=language
                    )

                    callback(
                        result
                    )

                except STTTimeoutError:
                    continue

                except Exception as exc:

                    logger.debug(
                        "Continuous STT cycle failed: %s",
                        exc,
                    )

                    result = TranscriptionResult(
                        text="",
                        success=False,
                        language=(
                            language
                            or self.config.language
                        ),
                        error=str(exc),
                    )

                    try:
                        callback(result)
                    except Exception:
                        logger.exception(
                            "STT callback failed."
                        )

                if interval > 0:

                    self._stop_event.wait(
                        interval
                    )

        finally:

            with self._lock:

                self._running = False

    def stop_continuous(self) -> None:

        self._stop_event.set()

        with self._lock:

            self._running = False

    def is_running(self) -> bool:

        return self._running

    # =========================================================================
    # TEXT PROCESSING
    # =========================================================================

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:

        if text is None:
            return ""

        text = str(
            text
        ).strip()

        text = " ".join(
            text.split()
        )

        return text

    # =========================================================================
    # LAST RESULT
    # =========================================================================

    def get_last_result(
        self,
    ) -> Optional[TranscriptionResult]:

        return self._last_result

    def get_last_text(self) -> str:

        result = self.get_last_result()

        if result is None:
            return ""

        return result.text

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "available": self.available,
            "running": self.is_running(),
            "language": self.config.language,
            "preferred_backend": (
                self.config.preferred_backend
            ),
            "available_backends": (
                self.get_available_backends()
            ),
            "sample_rate": (
                self.config.sample_rate
            ),
            "timeout": (
                self.config.timeout
            ),
            "phrase_time_limit": (
                self.config.phrase_time_limit
            ),
            "microphone_connected": (
                self.microphone is not None
            ),
        }

    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    def shutdown(self) -> None:

        self.stop_continuous()

        self.microphone = None

    close = shutdown


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================


SpeechRecognizer = SpeechToText


__all__ = [
    "STTConfig",
    "TranscriptionResult",
    "SpeechToTextError",
    "STTUnavailableError",
    "STTRecognitionError",
    "STTTimeoutError",
    "SpeechToText",
    "SpeechRecognizer",
]


