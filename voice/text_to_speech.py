"""
RENIX AI - Text To Speech

Natural voice output system for RENIX.

Responsibilities:
- Convert RENIX text responses into speech
- Support multiple TTS backends
- Voice selection
- Rate / volume / pitch configuration
- Synchronous and asynchronous speech
- Speech queue
- Stop/interruption support
- Multiple languages where the selected backend supports them
- Save generated speech to audio files when supported
- Provide status information to the rest of RENIX

The module is designed to work with optional TTS libraries and should not
prevent RENIX from starting if a particular backend is unavailable.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("RENIX.voice.text_to_speech")


# ============================================================================
# DATA TYPES
# ============================================================================


@dataclass
class TTSConfig:
    """RENIX text-to-speech configuration."""

    language: str = "en"

    voice: Optional[str] = None

    backend: str = "auto"

    rate: int = 175

    volume: float = 1.0

    pitch: Optional[float] = None

    enabled: bool = True

    asynchronous: bool = True

    queue_enabled: bool = True

    max_queue_size: int = 20

    interruptible: bool = True

    save_audio: bool = False

    output_directory: str = "data/audio"

    default_extension: str = ".wav"

    # Whether RENIX should automatically select a voice matching language.
    auto_select_voice: bool = True


@dataclass
class VoiceInfo:
    """Information about an installed TTS voice."""

    id: str

    name: str

    language: Optional[str] = None

    gender: Optional[str] = None

    provider: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "language": self.language,
            "gender": self.gender,
            "provider": self.provider,
            "metadata": self.metadata,
        }


@dataclass
class SpeechResult:
    """Result of a TTS operation."""

    text: str

    success: bool

    backend: Optional[str] = None

    voice: Optional[str] = None

    language: Optional[str] = None

    duration: Optional[float] = None

    output_file: Optional[str] = None

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
            "backend": self.backend,
            "voice": self.voice,
            "language": self.language,
            "duration": self.duration,
            "output_file": self.output_file,
            "error": self.error,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class TTSError(RuntimeError):
    """Base TTS error."""


class TTSUnavailableError(
    TTSError
):
    """Raised when no TTS backend is available."""


class TTSSynthesisError(
    TTSError
):
    """Raised when speech synthesis fails."""


class TTSPlaybackError(
    TTSError
):
    """Raised when audio playback fails."""


# ============================================================================
# MAIN CLASS
# ============================================================================


class TextToSpeech:
    """
    Main RENIX text-to-speech controller.

    Preferred architecture:

        RENIX response
              ↓
        TextToSpeech
              ↓
        selected backend
              ↓
        audio output

    Supported backends:

        pyttsx3
            Offline system TTS.

        gTTS
            Google Text-to-Speech service.

        auto
            Automatically selects the best available backend.
    """

    def __init__(
        self,
        config: Optional[TTSConfig] = None,
    ) -> None:

        self.config = (
            config
            or TTSConfig()
        )

        self._pyttsx3 = None
        self._engine = None

        self._gtts = None

        self._pygame = None

        self._lock = threading.RLock()

        self._speaking = False

        self._stop_event = threading.Event()

        self._worker_thread: Optional[
            threading.Thread
        ] = None

        self._speech_queue: queue.Queue = (
            queue.Queue(
                maxsize=max(
                    1,
                    self.config.max_queue_size,
                )
            )
        )

        self._last_result: Optional[
            SpeechResult
        ] = None

        self._initialize_backends()

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def _initialize_backends(
        self,
    ) -> None:

        try:

            import pyttsx3

            self._pyttsx3 = pyttsx3

            try:

                self._engine = (
                    pyttsx3.init()
                )

                self._configure_pyttsx3()

            except Exception:

                logger.exception(
                    "Failed initializing pyttsx3 engine."
                )

                self._engine = None

        except ImportError:

            logger.debug(
                "pyttsx3 is not installed."
            )

        try:

            from gtts import gTTS

            self._gtts = gTTS

        except ImportError:

            logger.debug(
                "gTTS is not installed."
            )

        try:

            import pygame

            self._pygame = pygame

        except ImportError:

            logger.debug(
                "pygame is not installed."
            )

    # =========================================================================
    # AVAILABILITY
    # =========================================================================

    @property
    def available(self) -> bool:

        return (
            self._engine is not None
            or self._gtts is not None
        )

    def is_available(self) -> bool:

        return self.available

    def require_available(self) -> None:

        if not self.available:

            raise TTSUnavailableError(
                "No TTS backend is available. "
                "Install pyttsx3 or gTTS."
            )

    def get_available_backends(
        self,
    ) -> list[str]:

        backends = []

        if self._engine is not None:
            backends.append(
                "pyttsx3"
            )

        if self._gtts is not None:
            backends.append(
                "gtts"
            )

        return backends

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def configure(
        self,
        *,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        backend: Optional[str] = None,
        rate: Optional[int] = None,
        volume: Optional[float] = None,
        pitch: Optional[float] = None,
    ) -> TTSConfig:

        if language is not None:

            language = str(
                language
            ).strip()

            if not language:

                raise ValueError(
                    "Language cannot be empty."
                )

            self.config.language = language

        if voice is not None:

            self.config.voice = str(
                voice
            )

            self._apply_voice()

        if backend is not None:

            backend = str(
                backend
            ).lower().strip()

            allowed = {
                "auto",
                "pyttsx3",
                "gtts",
            }

            if backend not in allowed:

                raise ValueError(
                    f"Unsupported TTS backend: {backend}"
                )

            self.config.backend = backend

        if rate is not None:

            rate = int(rate)

            if rate <= 0:

                raise ValueError(
                    "Speech rate must be positive."
                )

            self.config.rate = rate

            self._apply_rate()

        if volume is not None:

            volume = float(
                volume
            )

            if not 0.0 <= volume <= 1.0:

                raise ValueError(
                    "Volume must be between 0.0 and 1.0."
                )

            self.config.volume = volume

            self._apply_volume()

        if pitch is not None:

            self.config.pitch = float(
                pitch
            )

        return self.config

    # =========================================================================
    # PYTTSX3 CONFIGURATION
    # =========================================================================

    def _configure_pyttsx3(
        self,
    ) -> None:

        if self._engine is None:
            return

        self._apply_rate()
        self._apply_volume()
        self._apply_voice()

    def _apply_rate(self) -> None:

        if self._engine is None:
            return

        try:

            self._engine.setProperty(
                "rate",
                self.config.rate,
            )

        except Exception:

            logger.debug(
                "Could not set TTS rate.",
                exc_info=True,
            )

    def _apply_volume(self) -> None:

        if self._engine is None:
            return

        try:

            self._engine.setProperty(
                "volume",
                self.config.volume,
            )

        except Exception:

            logger.debug(
                "Could not set TTS volume.",
                exc_info=True,
            )

    def _apply_voice(self) -> None:

        if (
            self._engine is None
            or not self.config.voice
        ):
            return

        try:

            self._engine.setProperty(
                "voice",
                self.config.voice,
            )

        except Exception:

            logger.debug(
                "Could not set TTS voice.",
                exc_info=True,
            )

    # =========================================================================
    # VOICE MANAGEMENT
    # =========================================================================

    def list_voices(
        self,
    ) -> list[VoiceInfo]:

        voices: list[VoiceInfo] = []

        if self._engine is None:
            return voices

        try:

            installed = (
                self._engine.getProperty(
                    "voices"
                )
            )

            for voice in installed:

                voice_id = str(
                    getattr(
                        voice,
                        "id",
                        "",
                    )
                )

                name = str(
                    getattr(
                        voice,
                        "name",
                        voice_id,
                    )
                )

                languages = getattr(
                    voice,
                    "languages",
                    [],
                )

                language = None

                if languages:

                    try:

                        raw = languages[0]

                        if isinstance(
                            raw,
                            bytes,
                        ):

                            language = (
                                raw.decode(
                                    errors="ignore"
                                )
                            )

                        else:

                            language = str(
                                raw
                            )

                    except Exception:

                        language = None

                gender = getattr(
                    voice,
                    "gender",
                    None,
                )

                voices.append(
                    VoiceInfo(
                        id=voice_id,
                        name=name,
                        language=language,
                        gender=(
                            str(gender)
                            if gender
                            else None
                        ),
                        provider="pyttsx3",
                    )
                )

        except Exception:

            logger.exception(
                "Failed listing TTS voices."
            )

        return voices

    def set_voice(
        self,
        voice: str,
    ) -> bool:

        voice = str(
            voice
        ).strip()

        if not voice:

            raise ValueError(
                "Voice cannot be empty."
            )

        voices = self.list_voices()

        target = voice.lower()

        selected = None

        for item in voices:

            if (
                item.id.lower()
                == target
            ):

                selected = item
                break

            if (
                item.name.lower()
                == target
            ):

                selected = item
                break

        if selected is None:

            raise TTSError(
                f"TTS voice not found: {voice}"
            )

        self.config.voice = selected.id

        self._apply_voice()

        return True

    # =========================================================================
    # SPEAK
    # =========================================================================

    def speak(
        self,
        text: str,
        *,
        language: Optional[str] = None,
        backend: Optional[str] = None,
        asynchronous: Optional[bool] = None,
        save_file: Optional[str] = None,
        callback: Optional[
            Callable[[SpeechResult], None]
        ] = None,
    ) -> SpeechResult:

        self.require_available()

        text = self._clean_text(
            text
        )

        if not text:

            return SpeechResult(
                text="",
                success=False,
                error="Text is empty.",
            )

        language = (
            language
            or self.config.language
        )

        backend = (
            backend
            or self.config.backend
        )

        if asynchronous is None:

            asynchronous = (
                self.config.asynchronous
            )

        if (
            asynchronous
            and self.config.queue_enabled
        ):

            return self._queue_speech(
                text,
                language=language,
                backend=backend,
                save_file=save_file,
                callback=callback,
            )

        result = self._speak_sync(
            text,
            language=language,
            backend=backend,
            save_file=save_file,
        )

        if callback is not None:

            try:

                callback(result)

            except Exception:

                logger.exception(
                    "TTS callback failed."
                )

        return result

    def speak_sync(
        self,
        text: str,
        *,
        language: Optional[str] = None,
        backend: Optional[str] = None,
        save_file: Optional[str] = None,
    ) -> SpeechResult:

        return self._speak_sync(
            self._clean_text(text),
            language=(
                language
                or self.config.language
            ),
            backend=(
                backend
                or self.config.backend
            ),
            save_file=save_file,
        )

    def _speak_sync(
        self,
        text: str,
        *,
        language: str,
        backend: str,
        save_file: Optional[str],
    ) -> SpeechResult:

        started = time.monotonic()

        selected_backend = self._select_backend(
            backend
        )

        try:

            with self._lock:

                self._speaking = True

                self._stop_event.clear()

            if selected_backend == "pyttsx3":

                result = self._speak_pyttsx3(
                    text,
                    language=language,
                    save_file=save_file,
                )

            elif selected_backend == "gtts":

                result = self._speak_gtts(
                    text,
                    language=language,
                    save_file=save_file,
                )

            else:

                raise TTSUnavailableError(
                    f"Unsupported TTS backend: "
                    f"{selected_backend}"
                )

            result.duration = (
                time.monotonic()
                - started
            )

            self._last_result = result

            return result

        except Exception as exc:

            result = SpeechResult(
                text=text,
                success=False,
                backend=selected_backend,
                voice=self.config.voice,
                language=language,
                duration=(
                    time.monotonic()
                    - started
                ),
                error=str(exc),
            )

            self._last_result = result

            raise TTSSynthesisError(
                str(exc)
            ) from exc

        finally:

            with self._lock:

                self._speaking = False

    # =========================================================================
    # BACKEND SELECTION
    # =========================================================================

    def _select_backend(
        self,
        backend: str,
    ) -> str:

        backend = str(
            backend
        ).lower().strip()

        if backend == "auto":

            if self._engine is not None:
                return "pyttsx3"

            if self._gtts is not None:
                return "gtts"

            raise TTSUnavailableError(
                "No TTS backend available."
            )

        if backend == "pyttsx3":

            if self._engine is None:

                raise TTSUnavailableError(
                    "pyttsx3 backend is unavailable."
                )

            return backend

        if backend == "gtts":

            if self._gtts is None:

                raise TTSUnavailableError(
                    "gTTS backend is unavailable."
                )

            return backend

        raise TTSUnavailableError(
            f"Unknown TTS backend: {backend}"
        )

    # =========================================================================
    # PYTTSX3
    # =========================================================================

    def _speak_pyttsx3(
        self,
        text: str,
        *,
        language: str,
        save_file: Optional[str],
    ) -> SpeechResult:

        if self._engine is None:

            raise TTSUnavailableError(
                "pyttsx3 engine is unavailable."
            )

        if self._stop_event.is_set():

            return SpeechResult(
                text=text,
                success=False,
                backend="pyttsx3",
                voice=self.config.voice,
                language=language,
                error="Speech interrupted.",
            )

        self._configure_pyttsx3()

        output_file = None

        if save_file:

            output_file = str(
                Path(save_file)
                .expanduser()
                .resolve()
            )

            Path(
                output_file
            ).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._engine.save_to_file(
                text,
                output_file,
            )

        self._engine.say(
            text
        )

        self._engine.runAndWait()

        if self._stop_event.is_set():

            return SpeechResult(
                text=text,
                success=False,
                backend="pyttsx3",
                voice=self.config.voice,
                language=language,
                output_file=output_file,
                error="Speech interrupted.",
            )

        return SpeechResult(
            text=text,
            success=True,
            backend="pyttsx3",
            voice=self.config.voice,
            language=language,
            output_file=output_file,
        )

    # =========================================================================
    # gTTS
    # =========================================================================

    def _speak_gtts(
        self,
        text: str,
        *,
        language: str,
        save_file: Optional[str],
    ) -> SpeechResult:

        if self._gtts is None:

            raise TTSUnavailableError(
                "gTTS backend is unavailable."
            )

        output_file = (
            save_file
            or self._generate_output_path()
        )

        output_path = Path(
            output_file
        ).expanduser()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # gTTS language codes are usually short codes.
        gtts_language = (
            self._normalize_gtts_language(
                language
            )
        )

        tts = self._gtts(
            text=text,
            lang=gtts_language,
            slow=False,
        )

        tts.save(
            str(output_path)
        )

        self._play_file(
            str(output_path)
        )

        return SpeechResult(
            text=text,
            success=True,
            backend="gtts",
            voice=self.config.voice,
            language=language,
            output_file=str(
                output_path
            ),
        )

    # =========================================================================
    # QUEUE
    # =========================================================================

    def _queue_speech(
        self,
        text: str,
        *,
        language: str,
        backend: str,
        save_file: Optional[str],
        callback: Optional[
            Callable[[SpeechResult], None]
        ],
    ) -> SpeechResult:

        item = {
            "text": text,
            "language": language,
            "backend": backend,
            "save_file": save_file,
            "callback": callback,
        }

        try:

            self._speech_queue.put_nowait(
                item
            )

        except queue.Full:

            return SpeechResult(
                text=text,
                success=False,
                backend=backend,
                language=language,
                error="TTS queue is full.",
            )

        self._ensure_worker()

        # Asynchronous operations return immediately.
        return SpeechResult(
            text=text,
            success=True,
            backend=backend,
            language=language,
            metadata={
                "queued": True,
            },
        )

    def _ensure_worker(self) -> None:

        with self._lock:

            if (
                self._worker_thread is not None
                and self._worker_thread.is_alive()
            ):
                return

            self._worker_thread = threading.Thread(
                target=self._queue_worker,
                daemon=True,
                name="RENIX-TTS",
            )

            self._worker_thread.start()

    def _queue_worker(self) -> None:

        while True:

            try:

                item = (
                    self._speech_queue.get(
                        timeout=0.25
                    )
                )

            except queue.Empty:

                if self._speech_queue.empty():

                    with self._lock:

                        self._worker_thread = None

                    return

                continue

            try:

                result = self._speak_sync(
                    item["text"],
                    language=item[
                        "language"
                    ],
                    backend=item[
                        "backend"
                    ],
                    save_file=item[
                        "save_file"
                    ],
                )

                callback = item.get(
                    "callback"
                )

                if callback is not None:

                    try:

                        callback(result)

                    except Exception:

                        logger.exception(
                            "Queued TTS callback failed."
                        )

            except Exception as exc:

                logger.exception(
                    "Queued TTS failed: %s",
                    exc,
                )

                callback = item.get(
                    "callback"
                )

                if callback is not None:

                    try:

                        callback(
                            SpeechResult(
                                text=item[
                                    "text"
                                ],
                                success=False,
                                backend=item[
                                    "backend"
                                ],
                                language=item[
                                    "language"
                                ],
                                error=str(exc),
                            )
                        )

                    except Exception:

                        logger.exception(
                            "Queued TTS error callback failed."
                        )

            finally:

                self._speech_queue.task_done()

    # =========================================================================
    # PLAYBACK
    # =========================================================================

    def _play_file(
        self,
        path: str,
    ) -> None:

        if not os.path.exists(path):

            raise TTSPlaybackError(
                f"Audio file does not exist: {path}"
            )

        # Try pygame first.
        if self._pygame is not None:

            try:

                self._pygame.mixer.init()

                self._pygame.mixer.music.load(
                    path
                )

                self._pygame.mixer.music.set_volume(
                    self.config.volume
                )

                self._pygame.mixer.music.play()

                while (
                    self._pygame.mixer.music.get_busy()
                ):

                    if self._stop_event.is_set():

                        self._pygame.mixer.music.stop()

                        break

                    time.sleep(
                        0.03
                    )

                return

            except Exception:

                logger.debug(
                    "pygame playback failed; trying system playback.",
                    exc_info=True,
                )

        # Windows fallback.
        if os.name == "nt":

            try:

                import winsound

                flags = (
                    winsound.SND_FILENAME
                    | winsound.SND_ASYNC
                )

                winsound.PlaySound(
                    path,
                    flags,
                )

                # Wait approximately for WAV duration.
                duration = (
                    self._get_wav_duration(
                        path
                    )
                )

                end = (
                    time.monotonic()
                    + duration
                )

                while (
                    time.monotonic()
                    < end
                ):

                    if self._stop_event.is_set():
                        winsound.PlaySound(
                            None,
                            0,
                        )
                        break

                    time.sleep(
                        0.03
                    )

                return

            except Exception:

                logger.debug(
                    "Windows audio playback failed.",
                    exc_info=True,
                )

        raise TTSPlaybackError(
            "Unable to play generated audio."
        )

    @staticmethod
    def _get_wav_duration(
        path: str,
    ) -> float:

        try:

            with wave.open(
                path,
                "rb",
            ) as wav:

                frames = wav.getnframes()
                rate = wav.getframerate()

                if rate <= 0:
                    return 0.0

                return frames / rate

        except Exception:

            return 0.0

    # =========================================================================
    # FILE GENERATION
    # =========================================================================

    def _generate_output_path(
        self,
    ) -> str:

        directory = Path(
            self.config.output_directory
        ).expanduser()

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = int(
            time.time() * 1000
        )

        return str(
            directory
            / f"renix_tts_{timestamp}"
            f"{self.config.default_extension}"
        )

    # =========================================================================
    # INTERRUPTION
    # =========================================================================

    def stop(self) -> None:
        """
        Immediately request speech interruption.
        """

        self._stop_event.set()

        if self._engine is not None:

            try:

                self._engine.stop()

            except Exception:

                logger.debug(
                    "Failed stopping pyttsx3.",
                    exc_info=True,
                )

        if self._pygame is not None:

            try:

                if self._pygame.mixer.get_init():

                    self._pygame.mixer.music.stop()

            except Exception:

                logger.debug(
                    "Failed stopping pygame playback.",
                    exc_info=True,
                )

    interrupt = stop

    def clear_queue(self) -> int:

        removed = 0

        while True:

            try:

                self._speech_queue.get_nowait()

                self._speech_queue.task_done()

                removed += 1

            except queue.Empty:

                break

        return removed

    # =========================================================================
    # LANGUAGE
    # =========================================================================

    @staticmethod
    def _normalize_gtts_language(
        language: str,
    ) -> str:

        language = (
            str(language)
            .strip()
            .lower()
        )

        mapping = {
            "en-us": "en",
            "en-gb": "en",
            "en-in": "en",
            "hi-in": "hi",
            "gu-in": "gu",
            "mr-in": "mr",
            "ta-in": "ta",
            "te-in": "te",
            "kn-in": "kn",
            "bn-in": "bn",
            "pa-in": "pa",
        }

        return mapping.get(
            language,
            language.split("-")[0],
        )

    # =========================================================================
    # TEXT CLEANING
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

        # Avoid accidentally speaking huge repeated whitespace.
        text = " ".join(
            text.split()
        )

        return text

    # =========================================================================
    # STATE
    # =========================================================================

    def is_speaking(self) -> bool:

        return self._speaking

    def get_queue_size(self) -> int:

        return self._speech_queue.qsize()

    def get_last_result(
        self,
    ) -> Optional[SpeechResult]:

        return self._last_result

    def get_status(self) -> dict[str, Any]:

        return {
            "available": self.available,
            "speaking": self.is_speaking(),
            "queue_size": self.get_queue_size(),
            "language": self.config.language,
            "voice": self.config.voice,
            "backend": self.config.backend,
            "available_backends": (
                self.get_available_backends()
            ),
            "rate": self.config.rate,
            "volume": self.config.volume,
            "pitch": self.config.pitch,
        }

    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    def shutdown(self) -> None:

        self.stop()

        self.clear_queue()

        with self._lock:

            self._speaking = False

            self._worker_thread = None

        if self._pygame is not None:

            try:

                if self._pygame.mixer.get_init():

                    self._pygame.mixer.music.stop()
                    self._pygame.mixer.quit()

            except Exception:

                logger.debug(
                    "Failed shutting down pygame.",
                    exc_info=True,
                )

        self._engine = None

    close = shutdown


# ============================================================================
# ALIAS
# ============================================================================

TTS = TextToSpeech


__all__ = [
    "TTSConfig",
    "VoiceInfo",
    "SpeechResult",
    "TTSError",
    "TTSUnavailableError",
    "TTSSynthesisError",
    "TTSPlaybackError",
    "TextToSpeech",
    "TTS",
]


