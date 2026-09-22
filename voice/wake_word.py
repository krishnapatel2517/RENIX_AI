"""
RENIX AI - Wake Word Detection

Handles activation of RENIX through spoken wake phrases.

Examples:
    "Hey RENIX"
    "RENIX"
    "Wake up RENIX"

Responsibilities:
- Wake phrase detection
- Configurable wake phrases
- Case-insensitive matching
- Fuzzy matching for minor STT errors
- Detection from text produced by speech-to-text
- Optional continuous microphone monitoring
- Cooldown/debouncing
- Callback support
- Activation/deactivation state
- Thread-safe operation

The wake-word layer intentionally works on top of the STT system rather
than directly depending on a particular speech-recognition provider.

Architecture:

    Microphone
        ↓
    Speech To Text
        ↓
    WakeWordDetector
        ↓
    RENIX activation
        ↓
    Conversation / Command system
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger("RENIX.voice.wake_word")


# ============================================================================
# DATA TYPES
# ============================================================================


@dataclass
class WakeWordConfig:
    """Configuration for RENIX wake-word detection."""

    # Primary wake phrase.
    primary_phrase: str = "renix"

    # Additional phrases accepted by RENIX.
    aliases: list[str] = field(
        default_factory=lambda: [
            "hey renix",
            "okay renix",
            "ok renix",
            "wake up renix",
        ]
    )

    # Whether matching ignores capitalization.
    case_sensitive: bool = False

    # Enable fuzzy matching for speech-recognition mistakes.
    fuzzy_matching: bool = True

    # Minimum fuzzy similarity required.
    fuzzy_threshold: float = 0.82

    # Prevent repeated activations immediately after one another.
    cooldown: float = 1.5

    # Whether the detector should accept the wake phrase embedded
    # inside a longer sentence.
    allow_embedded_phrase: bool = True

    # Maximum number of words allowed around a wake phrase when using
    # fuzzy matching.
    fuzzy_window: int = 5

    # Continuous detector polling interval.
    polling_interval: float = 0.05


@dataclass
class WakeWordMatch:
    """Represents a detected wake phrase."""

    detected: bool

    phrase: Optional[str] = None

    matched_text: Optional[str] = None

    confidence: Optional[float] = None

    timestamp: float = field(
        default_factory=time.time
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "phrase": self.phrase,
            "matched_text": self.matched_text,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class WakeWordError(RuntimeError):
    """Base wake-word error."""


class WakeWordConfigurationError(
    WakeWordError
):
    """Invalid wake-word configuration."""


# ============================================================================
# MAIN CLASS
# ============================================================================


class WakeWordDetector:
    """
    RENIX wake-word detection engine.

    The detector can be used in two ways:

    1. Text mode

        detector.detect("Hey RENIX")

    2. Continuous microphone mode

        detector.start_listening()

    In continuous mode, the configured SpeechToText object is used to
    obtain text, which is then passed through this detector.
    """

    def __init__(
        self,
        config: Optional[WakeWordConfig] = None,
        speech_to_text: Optional[Any] = None,
    ) -> None:

        self.config = (
            config
            or WakeWordConfig()
        )

        self.speech_to_text = (
            speech_to_text
        )

        self._lock = threading.RLock()

        self._running = False

        self._active = False

        self._stop_event = threading.Event()

        self._worker_thread: Optional[
            threading.Thread
        ] = None

        self._last_match: Optional[
            WakeWordMatch
        ] = None

        self._last_activation = 0.0

        self._callbacks: list[
            Callable[[WakeWordMatch], None]
        ] = []

        self._validate_config()

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def _validate_config(self) -> None:

        if not self.config.primary_phrase.strip():

            raise WakeWordConfigurationError(
                "Primary wake phrase cannot be empty."
            )

        if not (
            0.0
            <= self.config.fuzzy_threshold
            <= 1.0
        ):

            raise WakeWordConfigurationError(
                "fuzzy_threshold must be between 0 and 1."
            )

        if self.config.cooldown < 0:

            raise WakeWordConfigurationError(
                "cooldown cannot be negative."
            )

        if self.config.fuzzy_window <= 0:

            raise WakeWordConfigurationError(
                "fuzzy_window must be positive."
            )

    def configure(
        self,
        *,
        primary_phrase: Optional[str] = None,
        aliases: Optional[
            Iterable[str]
        ] = None,
        fuzzy_matching: Optional[bool] = None,
        fuzzy_threshold: Optional[float] = None,
        cooldown: Optional[float] = None,
        allow_embedded_phrase: Optional[
            bool
        ] = None,
    ) -> WakeWordConfig:
        """Update wake-word settings."""

        if primary_phrase is not None:

            primary_phrase = str(
                primary_phrase
            ).strip()

            if not primary_phrase:

                raise WakeWordConfigurationError(
                    "Primary wake phrase cannot be empty."
                )

            self.config.primary_phrase = (
                primary_phrase
            )

        if aliases is not None:

            self.config.aliases = [
                str(item).strip()
                for item in aliases
                if str(item).strip()
            ]

        if fuzzy_matching is not None:

            self.config.fuzzy_matching = bool(
                fuzzy_matching
            )

        if fuzzy_threshold is not None:

            fuzzy_threshold = float(
                fuzzy_threshold
            )

            if not (
                0.0
                <= fuzzy_threshold
                <= 1.0
            ):

                raise WakeWordConfigurationError(
                    "fuzzy_threshold must be between 0 and 1."
                )

            self.config.fuzzy_threshold = (
                fuzzy_threshold
            )

        if cooldown is not None:

            cooldown = float(
                cooldown
            )

            if cooldown < 0:

                raise WakeWordConfigurationError(
                    "cooldown cannot be negative."
                )

            self.config.cooldown = cooldown

        if allow_embedded_phrase is not None:

            self.config.allow_embedded_phrase = bool(
                allow_embedded_phrase
            )

        self._validate_config()

        return self.config

    # =========================================================================
    # PHRASES
    # =========================================================================

    def get_phrases(self) -> list[str]:
        """
        Return all configured wake phrases without duplicates.
        """

        phrases: list[str] = []

        for phrase in [
            self.config.primary_phrase,
            *self.config.aliases,
        ]:

            normalized = self._normalize(
                phrase
            )

            if (
                normalized
                and normalized not in phrases
            ):

                phrases.append(
                    normalized
                )

        return phrases

    def add_phrase(
        self,
        phrase: str,
    ) -> None:

        phrase = str(
            phrase
        ).strip()

        if not phrase:

            raise WakeWordConfigurationError(
                "Wake phrase cannot be empty."
            )

        normalized = self._normalize(
            phrase
        )

        if normalized == self._normalize(
            self.config.primary_phrase
        ):

            return

        existing = {
            self._normalize(item)
            for item in self.config.aliases
        }

        if normalized not in existing:

            self.config.aliases.append(
                phrase
            )

    def remove_phrase(
        self,
        phrase: str,
    ) -> bool:

        target = self._normalize(
            phrase
        )

        for index, item in enumerate(
            self.config.aliases
        ):

            if (
                self._normalize(item)
                == target
            ):

                self.config.aliases.pop(
                    index
                )

                return True

        return False

    # =========================================================================
    # TEXT DETECTION
    # =========================================================================

    def detect(
        self,
        text: str,
    ) -> WakeWordMatch:
        """
        Check whether the supplied text contains a RENIX wake phrase.
        """

        if text is None:

            return WakeWordMatch(
                detected=False
            )

        text = str(
            text
        ).strip()

        if not text:

            return WakeWordMatch(
                detected=False
            )

        normalized_text = self._normalize(
            text
        )

        phrases = self.get_phrases()

        # ---------------------------------------------------------------
        # Exact matching
        # ---------------------------------------------------------------

        for phrase in phrases:

            if self._phrase_matches(
                normalized_text,
                phrase,
            ):

                confidence = 1.0

                match = WakeWordMatch(
                    detected=True,
                    phrase=phrase,
                    matched_text=text,
                    confidence=confidence,
                    metadata={
                        "method": "exact"
                    },
                )

                return self._process_match(
                    match
                )

        # ---------------------------------------------------------------
        # Fuzzy matching
        # ---------------------------------------------------------------

        if self.config.fuzzy_matching:

            fuzzy_match = (
                self._fuzzy_detect(
                    normalized_text,
                    phrases,
                )
            )

            if fuzzy_match is not None:

                return self._process_match(
                    fuzzy_match
                )

        return WakeWordMatch(
            detected=False,
            matched_text=text,
            metadata={
                "method": "none"
            },
        )

    def _phrase_matches(
        self,
        text: str,
        phrase: str,
    ) -> bool:

        if self.config.allow_embedded_phrase:

            # Word-boundary matching prevents "renix" from matching
            # arbitrary strings such as "renixed".
            pattern = (
                r"(?<!\w)"
                + re.escape(phrase)
                + r"(?!\w)"
            )

            flags = (
                0
                if self.config.case_sensitive
                else re.IGNORECASE
            )

            return re.search(
                pattern,
                text,
                flags,
            ) is not None

        return text == phrase

    # =========================================================================
    # FUZZY DETECTION
    # =========================================================================

    def _fuzzy_detect(
        self,
        text: str,
        phrases: list[str],
    ) -> Optional[WakeWordMatch]:

        words = text.split()

        if not words:

            return None

        best_phrase = None
        best_text = None
        best_score = 0.0

        for phrase in phrases:

            phrase_words = phrase.split()

            phrase_length = len(
                phrase_words
            )

            # Search windows around the expected phrase length.
            minimum = max(
                1,
                phrase_length - 1,
            )

            maximum = min(
                len(words),
                phrase_length
                + self.config.fuzzy_window,
            )

            for size in range(
                minimum,
                maximum + 1,
            ):

                for start in range(
                    0,
                    len(words) - size + 1,
                ):

                    candidate = " ".join(
                        words[
                            start : start + size
                        ]
                    )

                    score = (
                        SequenceMatcher(
                            None,
                            candidate,
                            phrase,
                        ).ratio()
                    )

                    if score > best_score:

                        best_score = score
                        best_phrase = phrase
                        best_text = candidate

        if (
            best_phrase is not None
            and best_score
            >= self.config.fuzzy_threshold
        ):

            return WakeWordMatch(
                detected=True,
                phrase=best_phrase,
                matched_text=best_text,
                confidence=best_score,
                metadata={
                    "method": "fuzzy"
                },
            )

        return None

    # =========================================================================
    # ACTIVATION / DEACTIVATION
    # =========================================================================

    def activate(
        self,
        *,
        force: bool = False,
    ) -> bool:
        """
        Activate RENIX.

        Returns False when activation is blocked by cooldown.
        """

        now = time.monotonic()

        with self._lock:

            if (
                not force
                and (
                    now
                    - self._last_activation
                    < self.config.cooldown
                )
            ):

                return False

            self._active = True

            self._last_activation = now

            return True

    def deactivate(self) -> None:

        with self._lock:

            self._active = False

    def is_active(self) -> bool:

        with self._lock:

            return self._active

    # =========================================================================
    # PROCESS DETECTION
    # =========================================================================

    def _process_match(
        self,
        match: WakeWordMatch,
    ) -> WakeWordMatch:

        if not match.detected:

            return match

        if not self.activate():

            return WakeWordMatch(
                detected=False,
                phrase=match.phrase,
                matched_text=match.matched_text,
                confidence=match.confidence,
                metadata={
                    **match.metadata,
                    "blocked_by_cooldown": True,
                },
            )

        self._last_match = match

        self._notify_callbacks(
            match
        )

        return match

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def add_callback(
        self,
        callback: Callable[
            [WakeWordMatch],
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
            [WakeWordMatch],
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

    def clear_callbacks(self) -> None:

        with self._lock:

            self._callbacks.clear()

    def _notify_callbacks(
        self,
        match: WakeWordMatch,
    ) -> None:

        with self._lock:

            callbacks = list(
                self._callbacks
            )

        for callback in callbacks:

            try:

                callback(
                    match
                )

            except Exception:

                logger.exception(
                    "Wake-word callback failed."
                )

    # =========================================================================
    # CONTINUOUS LISTENING
    # =========================================================================

    def start_listening(
        self,
        *,
        speech_to_text: Optional[Any] = None,
        callback: Optional[
            Callable[[WakeWordMatch], None]
        ] = None,
        language: Optional[str] = None,
    ) -> threading.Thread:
        """
        Start continuous wake-word monitoring.

        The STT object must expose:

            listen(...)

        and return a result object containing `.text`, or a string.
        """

        stt = (
            speech_to_text
            or self.speech_to_text
        )

        if stt is None:

            raise WakeWordError(
                "Speech-to-text system is not configured."
            )

        if not hasattr(
            stt,
            "listen",
        ):

            raise WakeWordError(
                "Configured STT object does not provide listen()."
            )

        if callback is not None:

            self.add_callback(
                callback
            )

        with self._lock:

            if self._running:

                raise WakeWordError(
                    "Wake-word listener is already running."
                )

            self._running = True

            self._stop_event.clear()

        self.speech_to_text = stt

        self._worker_thread = threading.Thread(
            target=self._listen_worker,
            args=(
                stt,
                language,
            ),
            daemon=True,
            name="RENIX-WakeWord",
        )

        self._worker_thread.start()

        return self._worker_thread

    def _listen_worker(
        self,
        stt: Any,
        language: Optional[str],
    ) -> None:

        try:

            while not self._stop_event.is_set():

                try:

                    result = stt.listen(
                        language=language
                    )

                    text = self._extract_text(
                        result
                    )

                    if text:

                        self.detect(
                            text
                        )

                except Exception as exc:

                    logger.debug(
                        "Wake-word listening cycle failed: %s",
                        exc,
                    )

                    self._stop_event.wait(
                        self.config.polling_interval
                    )

        finally:

            with self._lock:

                self._running = False

    @staticmethod
    def _extract_text(
        result: Any,
    ) -> str:

        if result is None:

            return ""

        if isinstance(
            result,
            str,
        ):

            return result

        if hasattr(
            result,
            "text",
        ):

            return str(
                result.text
                or ""
            )

        if isinstance(
            result,
            dict,
        ):

            return str(
                result.get(
                    "text",
                    "",
                )
                or ""
            )

        return str(
            result
        )

    def stop_listening(self) -> None:

        self._stop_event.set()

        with self._lock:

            self._running = False

    def is_listening(self) -> bool:

        with self._lock:

            return self._running

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _normalize(
        self,
        text: str,
    ) -> str:

        text = str(
            text
        ).strip()

        if not self.config.case_sensitive:

            text = text.lower()

        # Normalize common punctuation that STT systems may introduce.
        text = re.sub(
            r"[^\w\s'-]",
            " ",
            text,
            flags=re.UNICODE,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # =========================================================================
    # STATE
    # =========================================================================

    def get_last_match(
        self,
    ) -> Optional[WakeWordMatch]:

        return self._last_match

    def get_status(
        self,
    ) -> dict[str, Any]:

        return {
            "running": self.is_listening(),
            "active": self.is_active(),
            "primary_phrase": (
                self.config.primary_phrase
            ),
            "aliases": list(
                self.config.aliases
            ),
            "phrases": self.get_phrases(),
            "fuzzy_matching": (
                self.config.fuzzy_matching
            ),
            "fuzzy_threshold": (
                self.config.fuzzy_threshold
            ),
            "cooldown": (
                self.config.cooldown
            ),
            "speech_to_text_connected": (
                self.speech_to_text is not None
            ),
        }

    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    def shutdown(self) -> None:

        self.stop_listening()

        self.deactivate()

        self.clear_callbacks()

        self.speech_to_text = None

        with self._lock:

            self._last_match = None
            self._worker_thread = None

    close = shutdown


# ============================================================================
# ALIAS
# ============================================================================


WakeWord = WakeWordDetector


__all__ = [
    "WakeWordConfig",
    "WakeWordMatch",
    "WakeWordError",
    "WakeWordConfigurationError",
    "WakeWordDetector",
    "WakeWord",
]


