"""
RENIX AI - Voice Interruption / Barge-In Manager

Handles interruption of RENIX while it is speaking.

Responsibilities:
- Detect user speech while RENIX is speaking
- Stop/interrupt TTS safely
- Support barge-in
- Debounce repeated interruptions
- Track interruption state
- Coordinate microphone/STT/TTS
- Prevent accidental self-interruption
- Forward interrupted speech to conversation mode
- Provide callbacks/events
- Support temporary interruption suppression

This module does NOT implement:
- Microphone capture
- Speech-to-text
- Text-to-speech synthesis
- Wake-word detection
- Conversation lifecycle

Those systems are controlled by the corresponding voice modules.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


logger = logging.getLogger(
    "RENIX.voice.interruption"
)


# ============================================================================
# ENUMS
# ============================================================================


class InterruptionState(str, Enum):
    """Current interruption state."""

    IDLE = "idle"

    MONITORING = "monitoring"

    DETECTED = "detected"

    INTERRUPTING = "interrupting"

    INTERRUPTED = "interrupted"

    SUPPRESSED = "suppressed"

    ERROR = "error"


class InterruptionReason(str, Enum):
    """Reason an interruption occurred."""

    USER_SPEECH = "user_speech"

    USER_COMMAND = "user_command"

    BARGE_IN = "barge_in"

    EMERGENCY = "emergency"

    MANUAL = "manual"

    UNKNOWN = "unknown"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class InterruptionConfig:
    """Configuration for RENIX interruption handling."""

    enabled: bool = True

    barge_in_enabled: bool = True

    minimum_speech_duration_ms: int = 180

    minimum_confidence: float = 0.55

    debounce_ms: int = 500

    cooldown_ms: int = 700

    detection_window_ms: int = 250

    interrupt_only_when_speaking: bool = True

    allow_emergency_interruptions: bool = True

    suppress_during_startup: bool = True

    startup_suppression_seconds: float = 1.0

    suppress_after_tts_start_seconds: float = 0.25

    suppress_after_tts_stop_seconds: float = 0.15

    allow_partial_speech: bool = True

    minimum_text_length: int = 1

    auto_resume_listening: bool = True

    max_interruptions_per_minute: int = 30


@dataclass
class SpeechDetection:
    """
    Represents speech detected while RENIX is potentially speaking.
    """

    detection_id: str

    timestamp: float

    text: str = ""

    confidence: float = 0.0

    duration_ms: float = 0.0

    is_final: bool = False

    is_user_speech: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "detection_id": self.detection_id,
            "timestamp": self.timestamp,
            "text": self.text,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "is_final": self.is_final,
            "is_user_speech": self.is_user_speech,
            "metadata": self.metadata,
        }


@dataclass
class InterruptionEvent:
    """Information about a single interruption."""

    interruption_id: str

    timestamp: float

    reason: InterruptionReason

    speech: Optional[
        SpeechDetection
    ] = None

    handled: bool = False

    successful: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "interruption_id": (
                self.interruption_id
            ),
            "timestamp": self.timestamp,
            "reason": self.reason.value,
            "speech": (
                self.speech.to_dict()
                if self.speech
                else None
            ),
            "handled": self.handled,
            "successful": self.successful,
            "metadata": self.metadata,
        }


# ============================================================================
# EXCEPTIONS
# ============================================================================


class InterruptionError(
    RuntimeError
):
    """Base interruption exception."""


class InterruptionSuppressedError(
    InterruptionError
):
    """Raised when an interruption is intentionally suppressed."""


class InterruptionRateLimitError(
    InterruptionError
):
    """Raised when interruption rate exceeds configured limits."""


# ============================================================================
# INTERRUPTION MANAGER
# ============================================================================


class InterruptionManager:
    """
    RENIX interruption and barge-in controller.

    Typical flow:

        RENIX speaks
              ↓
        Microphone detects speech
              ↓
        STT / VAD reports speech
              ↓
        InterruptionManager
              ↓
        Stop TTS
              ↓
        ConversationMode
              ↓
        Process user's new command
    """

    def __init__(
        self,
        config: Optional[
            InterruptionConfig
        ] = None,
        *,
        is_speaking: Optional[
            Callable[[],
                    bool]
        ] = None,
        stop_speaking: Optional[
            Callable[[],
                    Any]
        ] = None,
        resume_listening: Optional[
            Callable[[],
                    Any]
        ] = None,
        on_interruption: Optional[
            Callable[
                [InterruptionEvent],
                Any,
            ]
        ] = None,
        on_speech_detected: Optional[
            Callable[
                [SpeechDetection],
                Any,
            ]
        ] = None,
    ) -> None:

        self.config = (
            config
            or InterruptionConfig()
        )

        self._lock = threading.RLock()

        self._state = (
            InterruptionState.IDLE
        )

        self._is_speaking = (
            is_speaking
        )

        self._stop_speaking = (
            stop_speaking
        )

        self._resume_listening = (
            resume_listening
        )

        self._on_interruption = (
            on_interruption
        )

        self._on_speech_detected = (
            on_speech_detected
        )

        self._callbacks: dict[
            str,
            list[Callable[..., Any]],
        ] = {}

        self._recent_interruptions: list[
            float
        ] = []

        self._last_detection_time = 0.0

        self._last_interrupt_time = 0.0

        self._tts_started_time = 0.0

        self._tts_stopped_time = 0.0

        self._startup_time = (
            time.monotonic()
        )

        self._suppressed_until = 0.0

        self._current_event: Optional[
            InterruptionEvent
        ] = None

        self._enabled = (
            self.config.enabled
        )

        self._validate_config()

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    def _validate_config(
        self,
    ) -> None:

        if (
            self.config.minimum_speech_duration_ms
            < 0
        ):

            raise ValueError(
                "minimum_speech_duration_ms "
                "cannot be negative."
            )

        if not (
            0.0
            <= self.config.minimum_confidence
            <= 1.0
        ):

            raise ValueError(
                "minimum_confidence must be "
                "between 0 and 1."
            )

        if (
            self.config.debounce_ms
            < 0
        ):

            raise ValueError(
                "debounce_ms cannot be negative."
            )

        if (
            self.config.cooldown_ms
            < 0
        ):

            raise ValueError(
                "cooldown_ms cannot be negative."
            )

        if (
            self.config.max_interruptions_per_minute
            <= 0
        ):

            raise ValueError(
                "max_interruptions_per_minute "
                "must be positive."
            )

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def state(
        self,
    ) -> InterruptionState:

        with self._lock:

            return self._state

    @property
    def enabled(
        self,
    ) -> bool:

        with self._lock:

            return self._enabled

    @property
    def active(
        self,
    ) -> bool:

        return (
            self.state
            in (
                InterruptionState.MONITORING,
                InterruptionState.DETECTED,
                InterruptionState.INTERRUPTING,
            )
        )

    # ========================================================================
    # STATE
    # ========================================================================

    def _set_state(
        self,
        state: InterruptionState,
    ) -> None:

        with self._lock:

            old_state = self._state

            if old_state == state:

                return

            self._state = state

        self._emit(
            "state_changed",
            {
                "old_state": (
                    old_state.value
                ),
                "new_state": (
                    state.value
                ),
            },
        )

    # ========================================================================
    # EVENTS
    # ========================================================================

    def register_callback(
        self,
        event_type: str,
        callback: Callable[..., Any],
    ) -> None:

        if not event_type:

            raise ValueError(
                "event_type cannot be empty."
            )

        if not callable(callback):

            raise TypeError(
                "callback must be callable."
            )

        with self._lock:

            self._callbacks.setdefault(
                event_type,
                [],
            ).append(
                callback
            )

    def unregister_callback(
        self,
        event_type: str,
        callback: Callable[..., Any],
    ) -> None:

        with self._lock:

            callbacks = self._callbacks.get(
                event_type,
                [],
            )

            if callback in callbacks:

                callbacks.remove(
                    callback
                )

    def _emit(
        self,
        event_type: str,
        data: Optional[
            dict[str, Any]
        ] = None,
    ) -> None:

        payload = {
            "event_type": event_type,
            "timestamp": time.time(),
            "data": data or {},
        }

        with self._lock:

            callbacks = list(
                self._callbacks.get(
                    event_type,
                    [],
                )
            )

        for callback in callbacks:

            try:

                callback(payload)

            except Exception:

                logger.exception(
                    "Interruption callback failed: %s",
                    event_type,
                )

    # ========================================================================
    # ENABLE / DISABLE
    # ========================================================================

    def enable(
        self,
    ) -> None:

        with self._lock:

            self._enabled = True

        self._set_state(
            InterruptionState.MONITORING
        )

        self._emit(
            "enabled"
        )

    def disable(
        self,
    ) -> None:

        with self._lock:

            self._enabled = False

        self._set_state(
            InterruptionState.IDLE
        )

        self._emit(
            "disabled"
        )

    # ========================================================================
    # MONITORING
    # ========================================================================

    def start_monitoring(
        self,
    ) -> None:

        if not self.enabled:

            return

        self._set_state(
            InterruptionState.MONITORING
        )

        self._emit(
            "monitoring_started"
        )

    def stop_monitoring(
        self,
    ) -> None:

        self._set_state(
            InterruptionState.IDLE
        )

        self._emit(
            "monitoring_stopped"
        )

    # ========================================================================
    # TTS STATE
    # ========================================================================

    def notify_tts_started(
        self,
    ) -> None:

        with self._lock:

            self._tts_started_time = (
                time.monotonic()
            )

        self._emit(
            "tts_started"
        )

    def notify_tts_stopped(
        self,
    ) -> None:

        with self._lock:

            self._tts_stopped_time = (
                time.monotonic()
            )

        self._emit(
            "tts_stopped"
        )

    def is_speaking(
        self,
    ) -> bool:

        callback = self._is_speaking

        if callback is None:

            return False

        try:

            return bool(
                callback()
            )

        except Exception:

            logger.exception(
                "is_speaking callback failed."
            )

            return False

    # ========================================================================
    # SUPPRESSION
    # ========================================================================

    def suppress(
        self,
        duration_seconds: float,
    ) -> None:

        if duration_seconds < 0:

            raise ValueError(
                "duration_seconds cannot be negative."
            )

        with self._lock:

            self._suppressed_until = max(
                self._suppressed_until,
                time.monotonic()
                + duration_seconds,
            )

        self._set_state(
            InterruptionState.SUPPRESSED
        )

        self._emit(
            "suppressed",
            {
                "duration_seconds": (
                    duration_seconds
                ),
            },
        )

    def clear_suppression(
        self,
    ) -> None:

        with self._lock:

            self._suppressed_until = 0.0

        if self.enabled:

            self._set_state(
                InterruptionState.MONITORING
            )

        self._emit(
            "suppression_cleared"
        )

    def is_suppressed(
        self,
    ) -> bool:

        now = time.monotonic()

        with self._lock:

            if (
                now
                < self._suppressed_until
            ):

                return True

            if (
                self.config.suppress_during_startup
                and (
                    now - self._startup_time
                    < self.config.startup_suppression_seconds
                )
            ):

                return True

            if (
                self._tts_started_time
                and (
                    now
                    - self._tts_started_time
                    < self.config.suppress_after_tts_start_seconds
                )
            ):

                return True

            return False

    # ========================================================================
    # RATE LIMITING
    # ========================================================================

    def _cleanup_rate_history(
        self,
        now: Optional[float] = None,
    ) -> None:

        now = (
            now
            if now is not None
            else time.monotonic()
        )

        cutoff = now - 60.0

        with self._lock:

            self._recent_interruptions = [
                timestamp
                for timestamp
                in self._recent_interruptions
                if timestamp >= cutoff
            ]

    def _rate_limit_reached(
        self,
    ) -> bool:

        now = time.monotonic()

        self._cleanup_rate_history(
            now
        )

        with self._lock:

            return (
                len(
                    self._recent_interruptions
                )
                >= self.config.max_interruptions_per_minute
            )

    # ========================================================================
    # SPEECH VALIDATION
    # ========================================================================

    def _validate_detection(
        self,
        detection: SpeechDetection,
    ) -> bool:

        if not detection.is_user_speech:

            return False

        if (
            detection.confidence
            < self.config.minimum_confidence
        ):

            return False

        if (
            detection.duration_ms
            < self.config.minimum_speech_duration_ms
        ):

            return False

        if (
            self.config.minimum_text_length
            > 0
            and len(
                detection.text.strip()
            )
            < self.config.minimum_text_length
        ):

            if not self.config.allow_partial_speech:

                return False

        return True

    # ========================================================================
    # SPEECH DETECTION
    # ========================================================================

    def detect_speech(
        self,
        text: str = "",
        *,
        confidence: float = 1.0,
        duration_ms: float = 0.0,
        is_final: bool = False,
        is_user_speech: bool = True,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Optional[
        SpeechDetection
    ]:

        detection = SpeechDetection(
            detection_id=str(
                uuid.uuid4()
            ),
            timestamp=time.time(),
            text=str(
                text or ""
            ).strip(),
            confidence=float(
                confidence
            ),
            duration_ms=float(
                duration_ms
            ),
            is_final=bool(
                is_final
            ),
            is_user_speech=bool(
                is_user_speech
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        with self._lock:

            self._last_detection_time = (
                time.monotonic()
            )

        callback = (
            self._on_speech_detected
        )

        if callback is not None:

            try:

                callback(detection)

            except Exception:

                logger.exception(
                    "Speech detection callback failed."
                )

        self._emit(
            "speech_detected",
            detection.to_dict(),
        )

        if not self.enabled:

            return None

        if not self._validate_detection(
            detection
        ):

            self._emit(
                "speech_rejected",
                detection.to_dict(),
            )

            return None

        return detection

    # ========================================================================
    # INTERRUPTION DETECTION
    # ========================================================================

    def should_interrupt(
        self,
        detection: SpeechDetection,
    ) -> bool:

        if not self.enabled:

            return False

        if not self.config.barge_in_enabled:

            return False

        if self.is_suppressed():

            return False

        if (
            self.config.interrupt_only_when_speaking
            and not self.is_speaking()
        ):

            return False

        if not self._validate_detection(
            detection
        ):

            return False

        now = time.monotonic()

        with self._lock:

            debounce_seconds = (
                self.config.debounce_ms
                / 1000.0
            )

            cooldown_seconds = (
                self.config.cooldown_ms
                / 1000.0
            )

            if (
                now
                - self._last_detection_time
                < debounce_seconds
            ):

                # The detection itself updates
                # _last_detection_time, so only
                # reject when another detection
                # arrived immediately.
                pass

            if (
                now
                - self._last_interrupt_time
                < cooldown_seconds
            ):

                return False

        if self._rate_limit_reached():

            self._emit(
                "rate_limit_reached"
            )

            return False

        return True

    # ========================================================================
    # PROCESS DETECTION
    # ========================================================================

    def process_speech(
        self,
        text: str = "",
        *,
        confidence: float = 1.0,
        duration_ms: float = 0.0,
        is_final: bool = False,
        is_user_speech: bool = True,
        reason: InterruptionReason = (
            InterruptionReason.USER_SPEECH
        ),
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Optional[
        InterruptionEvent
    ]:

        detection = self.detect_speech(
            text,
            confidence=confidence,
            duration_ms=duration_ms,
            is_final=is_final,
            is_user_speech=is_user_speech,
            metadata=metadata,
        )

        if detection is None:

            return None

        if not self.should_interrupt(
            detection
        ):

            return None

        return self.interrupt(
            detection=detection,
            reason=reason,
            metadata=metadata,
        )

    # ========================================================================
    # EXECUTE INTERRUPTION
    # ========================================================================

    def interrupt(
        self,
        *,
        detection: Optional[
            SpeechDetection
        ] = None,
        reason: InterruptionReason = (
            InterruptionReason.USER_SPEECH
        ),
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Optional[
        InterruptionEvent
    ]:

        if not self.enabled:

            return None

        if (
            reason
            == InterruptionReason.EMERGENCY
            and self.config.allow_emergency_interruptions
        ):

            pass

        elif self.is_suppressed():

            return None

        if self._rate_limit_reached():

            return None

        event = InterruptionEvent(
            interruption_id=str(
                uuid.uuid4()
            ),
            timestamp=time.time(),
            reason=reason,
            speech=detection,
            metadata=dict(
                metadata or {}
            ),
        )

        with self._lock:

            self._current_event = event

            self._last_interrupt_time = (
                time.monotonic()
            )

            self._recent_interruptions.append(
                self._last_interrupt_time
            )

        self._set_state(
            InterruptionState.INTERRUPTING
        )

        self._emit(
            "interruption_detected",
            event.to_dict(),
        )

        success = False

        try:

            callback = (
                self._stop_speaking
            )

            if callback is not None:

                result = callback()

                if result is False:

                    success = False

                else:

                    success = True

            else:

                # If there is no TTS callback,
                # still mark the event so the
                # conversation engine can handle it.
                success = True

        except Exception as exc:

            logger.exception(
                "Failed to stop RENIX speech."
            )

            event.metadata[
                "error"
            ] = str(exc)

            self._set_state(
                InterruptionState.ERROR
            )

        event.handled = True

        event.successful = success

        with self._lock:

            self._tts_stopped_time = (
                time.monotonic()
            )

        if success:

            self._set_state(
                InterruptionState.INTERRUPTED
            )

            self._emit(
                "interruption_completed",
                event.to_dict(),
            )

            callback = (
                self._on_interruption
            )

            if callback is not None:

                try:

                    callback(event)

                except Exception:

                    logger.exception(
                        "Interruption callback failed."
                    )

            if (
                self.config.auto_resume_listening
                and self._resume_listening
            ):

                try:

                    self._resume_listening()

                except Exception:

                    logger.exception(
                        "Failed to resume listening."
                    )

        else:

            self._emit(
                "interruption_failed",
                event.to_dict(),
            )

        return event

    # ========================================================================
    # EMERGENCY INTERRUPTION
    # ========================================================================

    def emergency_interrupt(
        self,
        text: str = "",
        *,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Optional[
        InterruptionEvent
    ]:

        if not self.config.allow_emergency_interruptions:

            return None

        detection = SpeechDetection(
            detection_id=str(
                uuid.uuid4()
            ),
            timestamp=time.time(),
            text=str(
                text or ""
            ).strip(),
            confidence=1.0,
            duration_ms=1000.0,
            is_final=True,
            is_user_speech=True,
            metadata=dict(
                metadata or {}
            ),
        )

        return self.interrupt(
            detection=detection,
            reason=InterruptionReason.EMERGENCY,
            metadata={
                "emergency": True,
                **dict(
                    metadata or {}
                ),
            },
        )

    # ========================================================================
    # MANUAL INTERRUPTION
    # ========================================================================

    def manual_interrupt(
        self,
        *,
        reason: str = "manual",
    ) -> Optional[
        InterruptionEvent
    ]:

        return self.interrupt(
            reason=InterruptionReason.MANUAL,
            metadata={
                "manual_reason": reason,
            },
        )

    # ========================================================================
    # COORDINATION WITH CONVERSATION MODE
    # ========================================================================

    def attach_conversation_mode(
        self,
        conversation_mode: Any,
    ) -> None:
        """
        Connect interruption handling to RENIX's
        conversation_mode.py.

        The object is intentionally duck-typed so
        this module remains independent from the
        conversation implementation.
        """

        if conversation_mode is None:

            raise ValueError(
                "conversation_mode cannot be None."
            )

        if hasattr(
            conversation_mode,
            "interrupt",
        ):

            def handle_event(
                event: InterruptionEvent,
            ) -> None:

                try:

                    conversation_mode.interrupt(
                        reason=(
                            event.reason.value
                        )
                    )

                except Exception:

                    logger.exception(
                        "Conversation mode interruption failed."
                    )

            self._on_interruption = (
                handle_event
            )

        if hasattr(
            conversation_mode,
            "set_speech_output_active",
        ):

            self._is_speaking = (
                lambda: bool(
                    conversation_mode.speaking
                )
            )

    # ========================================================================
    # RESET
    # ========================================================================

    def reset(
        self,
    ) -> None:

        with self._lock:

            self._state = (
                InterruptionState.IDLE
            )

            self._recent_interruptions.clear()

            self._last_detection_time = 0.0

            self._last_interrupt_time = 0.0

            self._tts_started_time = 0.0

            self._tts_stopped_time = 0.0

            self._suppressed_until = 0.0

            self._current_event = None

            self._startup_time = (
                time.monotonic()
            )

        self._emit(
            "reset"
        )

    # ========================================================================
    # STATUS
    # ========================================================================

    def get_status(
        self,
    ) -> dict[str, Any]:

        now = time.monotonic()

        self._cleanup_rate_history(
            now
        )

        with self._lock:

            return {
                "enabled": self._enabled,
                "state": self._state.value,
                "active": self.active,
                "is_speaking": self.is_speaking(),
                "suppressed": self.is_suppressed(),
                "recent_interruptions": len(
                    self._recent_interruptions
                ),
                "max_interruptions_per_minute": (
                    self.config.max_interruptions_per_minute
                ),
                "last_detection": (
                    self._last_detection_time
                ),
                "last_interruption": (
                    self._last_interrupt_time
                ),
                "current_event": (
                    self._current_event.to_dict()
                    if self._current_event
                    else None
                ),
            }

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    def shutdown(
        self,
    ) -> None:

        try:

            if self.is_speaking():

                callback = (
                    self._stop_speaking
                )

                if callback is not None:

                    try:

                        callback()

                    except Exception:

                        logger.exception(
                            "Failed to stop speech during shutdown."
                        )

        finally:

            self.disable()

            self.reset()


# ============================================================================
# DEFAULT SINGLETON
# ============================================================================


_default_manager: Optional[
    InterruptionManager
] = None

_default_lock = threading.RLock()


def get_interruption_manager() -> InterruptionManager:
    """Return the process-wide RENIX interruption manager."""

    global _default_manager

    with _default_lock:

        if _default_manager is None:

            _default_manager = (
                InterruptionManager()
            )

        return _default_manager


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def enable_interruption() -> None:

    get_interruption_manager().enable()


def disable_interruption() -> None:

    get_interruption_manager().disable()


def interrupt_renix(
    *,
    reason: InterruptionReason = (
        InterruptionReason.MANUAL
    ),
) -> Optional[
    InterruptionEvent
]:

    return (
        get_interruption_manager()
        .interrupt(
            reason=reason
        )
    )


def notify_speech(
    text: str,
    *,
    confidence: float = 1.0,
    duration_ms: float = 0.0,
    is_final: bool = False,
) -> Optional[
    InterruptionEvent
]:

    return (
        get_interruption_manager()
        .process_speech(
            text,
            confidence=confidence,
            duration_ms=duration_ms,
            is_final=is_final,
        )
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "InterruptionState",
    "InterruptionReason",
    "InterruptionConfig",
    "SpeechDetection",
    "InterruptionEvent",
    "InterruptionError",
    "InterruptionSuppressedError",
    "InterruptionRateLimitError",
    "InterruptionManager",
    "get_interruption_manager",
    "enable_interruption",
    "disable_interruption",
    "interrupt_renix",
    "notify_speech",
]


