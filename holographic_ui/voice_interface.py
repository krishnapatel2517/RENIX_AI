"""
RENIX Holographic UI - Voice Interface
======================================

Connects the RENIX voice system with the holographic UI.

Responsibilities:
    - Receive voice/transcription events
    - Display live transcription
    - Display assistant responses
    - Detect command state
    - Control voice-driven UI modes
    - Show listening/thinking/speaking indicators
    - Connect voice commands to UI callbacks

The actual microphone, STT, TTS, wake-word and conversation systems
remain inside the voice/ package.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


logger = logging.getLogger(
    "RENIX.holographic_ui.voice_interface"
)


# ============================================================================
# ENUMS
# ============================================================================


class VoiceState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class VoiceEvent:
    """Normalized voice event."""

    event_type: str

    text: str = ""

    confidence: float = 1.0

    language: Optional[str] = None

    is_final: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    timestamp: float = field(
        default_factory=time.time
    )


@dataclass
class VoiceUIState:
    """Current state displayed by the holographic UI."""

    state: VoiceState = VoiceState.IDLE

    transcript: str = ""

    response: str = ""

    language: Optional[str] = None

    confidence: float = 0.0

    listening: bool = False

    processing: bool = False

    speaking: bool = False

    muted: bool = False

    wake_detected: bool = False

    error: Optional[str] = None

    timestamp: float = field(
        default_factory=time.time
    )


# ============================================================================
# VOICE INTERFACE
# ============================================================================


class VoiceInterface:
    """
    Bridge between RENIX voice services and holographic UI.
    """

    def __init__(
        self,
        *,
        voice_manager: Any = None,
        speech_to_text: Any = None,
        text_to_speech: Any = None,
        conversation_mode: Any = None,
    ) -> None:

        self.voice_manager = voice_manager

        self.speech_to_text = speech_to_text

        self.text_to_speech = text_to_speech

        self.conversation_mode = (
            conversation_mode
        )

        self.state = VoiceUIState()

        self._callbacks: dict[
            str,
            list[Callable[..., None]],
        ] = {}

        self._lock = threading.RLock()

        self._running = False

        self._last_event_time = 0.0

        logger.debug(
            "Voice interface created"
        )

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self) -> None:

        with self._lock:

            if self._running:

                return

            self._running = True

        self._connect_voice_services()

        self._emit(
            "started",
            self,
        )

        logger.info(
            "Voice interface started"
        )

    def stop(self) -> None:

        with self._lock:

            if not self._running:

                return

            self._running = False

            self.state = VoiceUIState()

        self._disconnect_voice_services()

        self._emit(
            "stopped",
            self,
        )

        logger.info(
            "Voice interface stopped"
        )

    # ========================================================================
    # SERVICE CONNECTION
    # ========================================================================

    def _connect_voice_services(
        self,
    ) -> None:

        services = [
            self.voice_manager,
            self.speech_to_text,
            self.text_to_speech,
            self.conversation_mode,
        ]

        for service in services:

            if service is None:

                continue

            try:

                if hasattr(service, "on"):

                    service.on(
                        "voice_event",
                        self.handle_event,
                    )

                    service.on(
                        "transcription",
                        self.handle_transcription,
                    )

                    service.on(
                        "response",
                        self.handle_response,
                    )

                    service.on(
                        "state",
                        self.handle_state,
                    )

            except Exception:

                logger.exception(
                    "Failed connecting voice service"
                )

    def _disconnect_voice_services(
        self,
    ) -> None:

        services = [
            self.voice_manager,
            self.speech_to_text,
            self.text_to_speech,
            self.conversation_mode,
        ]

        for service in services:

            if service is None:

                continue

            try:

                if hasattr(service, "off"):

                    service.off(
                        "voice_event",
                        self.handle_event,
                    )

                    service.off(
                        "transcription",
                        self.handle_transcription,
                    )

                    service.off(
                        "response",
                        self.handle_response,
                    )

                    service.off(
                        "state",
                        self.handle_state,
                    )

            except Exception:

                logger.exception(
                    "Failed disconnecting voice service"
                )

    # ========================================================================
    # EVENT HANDLING
    # ========================================================================

    def handle_event(
        self,
        event: VoiceEvent
        | dict[str, Any]
        | Any,
    ) -> None:

        if not self._running:

            return

        normalized = self._normalize_event(
            event
        )

        if normalized is None:

            return

        self._last_event_time = (
            normalized.timestamp
        )

        event_type = (
            normalized.event_type
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        handlers = {
            "wake": self.handle_wake,
            "wake_word": self.handle_wake,
            "listening": self.handle_listening,
            "transcription": self.handle_transcription,
            "speech": self.handle_transcription,
            "processing": self.handle_processing,
            "thinking": self.handle_processing,
            "response": self.handle_response,
            "speaking": self.handle_speaking,
            "interrupt": self.handle_interrupt,
            "interrupted": self.handle_interrupt,
            "error": self.handle_error,
            "idle": self.handle_idle,
            "mute": self.handle_mute,
            "unmute": self.handle_unmute,
        }

        handler = handlers.get(
            event_type
        )

        if handler is not None:

            try:

                handler(normalized)

            except Exception:

                logger.exception(
                    "Voice event handler failed: %s",
                    event_type,
                )

        else:

            self._emit(
                "unknown_event",
                normalized,
            )

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    def _normalize_event(
        self,
        event: VoiceEvent
        | dict[str, Any]
        | Any,
    ) -> Optional[VoiceEvent]:

        if isinstance(
            event,
            VoiceEvent,
        ):

            return event

        if isinstance(
            event,
            dict,
        ):

            return VoiceEvent(
                event_type=str(
                    event.get(
                        "event_type",
                        event.get(
                            "type",
                            "",
                        ),
                    )
                ),
                text=str(
                    event.get(
                        "text",
                        "",
                    )
                ),
                confidence=float(
                    event.get(
                        "confidence",
                        1.0,
                    )
                ),
                language=event.get(
                    "language"
                ),
                is_final=bool(
                    event.get(
                        "is_final",
                        True,
                    )
                ),
                metadata=dict(
                    event.get(
                        "metadata",
                        {},
                    )
                ),
                timestamp=float(
                    event.get(
                        "timestamp",
                        time.time(),
                    )
                ),
            )

        event_type = getattr(
            event,
            "event_type",
            getattr(
                event,
                "type",
                None,
            ),
        )

        if event_type is None:

            return None

        return VoiceEvent(
            event_type=str(
                event_type
            ),
            text=str(
                getattr(
                    event,
                    "text",
                    "",
                )
            ),
            confidence=float(
                getattr(
                    event,
                    "confidence",
                    1.0,
                )
            ),
            language=getattr(
                event,
                "language",
                None,
            ),
            is_final=bool(
                getattr(
                    event,
                    "is_final",
                    True,
                )
            ),
            timestamp=float(
                getattr(
                    event,
                    "timestamp",
                    time.time(),
                )
            ),
        )

    # ========================================================================
    # WAKE WORD
    # ========================================================================

    def handle_wake(
        self,
        event: VoiceEvent,
    ) -> None:

        with self._lock:

            self.state.wake_detected = True

            self.state.state = (
                VoiceState.LISTENING
            )

            self.state.listening = True

            self.state.processing = False

            self.state.speaking = False

            self.state.error = None

            self.state.timestamp = time.time()

        self._emit(
            "wake_detected",
            event,
            self.get_state(),
        )

        self._emit(
            "ui_state_changed",
            self.get_state(),
        )

    # ========================================================================
    # LISTENING
    # ========================================================================

    def handle_listening(
        self,
        event: Optional[VoiceEvent] = None,
    ) -> None:

        with self._lock:

            self.state.state = (
                VoiceState.LISTENING
            )

            self.state.listening = True

            self.state.processing = False

            self.state.speaking = False

            self.state.error = None

            self.state.timestamp = time.time()

        self._emit(
            "listening_started",
            event,
            self.get_state(),
        )

        self._emit(
            "ui_state_changed",
            self.get_state(),
        )

    # ========================================================================
    # TRANSCRIPTION
    # ========================================================================

    def handle_transcription(
        self,
        event: VoiceEvent
        | dict[str, Any]
        | str,
    ) -> None:

        if isinstance(
            event,
            str,
        ):

            event = VoiceEvent(
                event_type="transcription",
                text=event,
            )

        else:

            event = self._normalize_event(
                event
            )

        if event is None:

            return

        with self._lock:

            self.state.transcript = (
                event.text
            )

            self.state.language = (
                event.language
            )

            self.state.confidence = (
                event.confidence
            )

            self.state.listening = (
                not event.is_final
            )

            self.state.timestamp = time.time()

        self._emit(
            "transcription",
            event.text,
            event,
            self.get_state(),
        )

        self._emit(
            "transcript_changed",
            event.text,
        )

        self._emit(
            "ui_state_changed",
            self.get_state(),
        )

        if event.is_final:

            self.handle_processing(
                event
            )

    # ========================================================================
    # PROCESSING
    # ========================================================================

    def handle_processing(
        self,
        event: Optional[VoiceEvent] = None,
    ) -> None:

        with self._lock:

            self.state.state = (
                VoiceState.PROCESSING
            )

            self.state.listening = False

            self.state.processing = True

            self.state.speaking = False

            self.state.error = None

            self.state.timestamp = time.time()

        self._emit(
            "processing_started",
            event,
            self.get_state(),
        )

        self._emit(
            "ui_state_changed",
            self.get_state(),
        )

    # ========================================================================
    # RESPONSE
    # ========================================================================

    def handle_response(
        self,
        event: VoiceEvent
        | dict[str, Any]
        | str,
    ) -> None:

        if isinstance(
            event,
            str,
        ):

            event = VoiceEvent(
                event_type="response",
                text=event,
            )

        else:

            event = self._normalize_event(
                event
            )

        if event is None:

            return

        with self._lock:

            self.state.response = (
                event.text
            )

            self.state.processing = False

            self.state.timestamp = time.time()

        self._emit(
            "response",
            event.text,
            event,
            self.get_state(),
        )

        self._emit(
            "response_changed",
            event.text,
        )

        self._emit(
            "ui_state_changed",
            self.get_state(),
        )

    # ========================================================================
    # SPEAKING
    # ========================================================================

    def handle_speaking(
        self,
        event: Optional[VoiceEvent] = None,
    ) -> None:

        with self._lock:

            self.state.state = (
                VoiceState.SPEAKING
            )

            self.state.listening = False

            self.state.processing = False

            self.state.speaking = True

            self.state.error = None

            self.state.timestamp = time.time()

        self._emit(
            "speaking_started",
            event,
            self.get_state(),
        )

        self._emit(
            "ui_state_changed",
            self.get_state(),
        )

    # ========================================================================
    # INTERRUPTION
    # ========================================================================

    def handle_interrupt(
        self,
        event: Optional[VoiceEvent] = None,
    ) -> None:

        with self._lock:

            self.state.state = (
                VoiceState.INTERRUPTED
            )

            self.state.speaking = False

            self.state.processing = False

            self.state.timestamp = time.time()

        self._emit(
            "interrupted",
            event,
            self.get_state(),
        )

        self._emit(
            "ui_state_changed",
            self.get_state(),
        )

    # ========================================================================
    # IDLE
    # ========================================================================

    def handle_idle(
        self,
        event: Optional[VoiceEvent] = None,
    ) -> None:

        with self._lock:

            self.state.state = (
                VoiceState.IDLE
            )

            self.state.listening = False

            self.state.processing = False

            self.state.speaking = False

            self.state.wake_detected = False

            self.state.error = None

            self.state.timestamp = time.time()

        self._emit(
            "idle",
            event,
            self.get_state(),
        )

        self._emit(
            "ui_state_changed",
            self.get_state(),
        )

    # ========================================================================
    # MUTE
    # ========================================================================

    def handle_mute(
        self,
        event: Optional[VoiceEvent] = None,
    ) -> None:

        with self._lock:

            self.state.muted = True

            self.state.listening = False

            self.state.timestamp = time.time()

        self._emit(
            "muted",
            event,
        )

        self._emit(
            "ui_state_changed",
            self.get_state(),
        )

    def handle_unmute(
        self,
        event: Optional[VoiceEvent] = None,
    ) -> None:

        with self._lock:

            self.state.muted = False

            self.state.timestamp = time.time()

        self._emit(
            "unmuted",
            event,
        )

        self._emit(
            "ui_state_changed",
            self.get_state(),
        )

    # ========================================================================
    # ERROR
    # ========================================================================

    def handle_error(
        self,
        event: VoiceEvent
        | dict[str, Any]
        | str,
    ) -> None:

        if isinstance(
            event,
            str,
        ):

            message = event

        else:

            normalized = self._normalize_event(
                event
            )

            if normalized is None:

                message = "Unknown voice error"

            else:

                message = normalized.text

        with self._lock:

            self.state.state = (
                VoiceState.ERROR
            )

            self.state.listening = False

            self.state.processing = False

            self.state.speaking = False

            self.state.error = message

            self.state.timestamp = time.time()

        self._emit(
            "error",
            message,
            self.get_state(),
        )

        self._emit(
            "ui_state_changed",
            self.get_state(),
        )

    # ========================================================================
    # COMMANDS FROM UI
    # ========================================================================

    def submit_command(
        self,
        text: str,
    ) -> bool:

        text = str(text).strip()

        if not text:

            return False

        self.handle_transcription(
            VoiceEvent(
                event_type="transcription",
                text=text,
                is_final=True,
            )
        )

        self._emit(
            "command_submitted",
            text,
        )

        return True

    def clear_transcript(self) -> None:

        with self._lock:

            self.state.transcript = ""

        self._emit(
            "transcript_changed",
            "",
        )

        self._emit(
            "ui_state_changed",
            self.get_state(),
        )

    def clear_response(self) -> None:

        with self._lock:

            self.state.response = ""

        self._emit(
            "response_changed",
            "",
        )

        self._emit(
            "ui_state_changed",
            self.get_state(),
        )

    # ========================================================================
    # STATE
    # ========================================================================

    def get_state(self) -> VoiceUIState:

        with self._lock:

            return VoiceUIState(
                **vars(self.state)
            )

    @property
    def running(self) -> bool:

        with self._lock:

            return self._running

    # ========================================================================
    # CALLBACK SYSTEM
    # ========================================================================

    def on(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:

        if not event:

            raise ValueError(
                "event cannot be empty"
            )

        if not callable(callback):

            raise TypeError(
                "callback must be callable"
            )

        with self._lock:

            self._callbacks.setdefault(
                event,
                [],
            ).append(callback)

    def off(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:

        with self._lock:

            callbacks = self._callbacks.get(
                event,
                [],
            )

            if callback in callbacks:

                callbacks.remove(
                    callback
                )

    def _emit(
        self,
        event: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        with self._lock:

            callbacks = list(
                self._callbacks.get(
                    event,
                    [],
                )
            )

        for callback in callbacks:

            try:

                callback(
                    *args,
                    **kwargs,
                )

            except Exception:

                logger.exception(
                    "Voice UI callback failed: %s",
                    event,
                )

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def to_dict(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "running": self._running,
                "state": (
                    self.state.state.value
                ),
                "transcript": (
                    self.state.transcript
                ),
                "response": (
                    self.state.response
                ),
                "language": (
                    self.state.language
                ),
                "confidence": (
                    self.state.confidence
                ),
                "listening": (
                    self.state.listening
                ),
                "processing": (
                    self.state.processing
                ),
                "speaking": (
                    self.state.speaking
                ),
                "muted": (
                    self.state.muted
                ),
                "wake_detected": (
                    self.state.wake_detected
                ),
                "error": (
                    self.state.error
                ),
            }


__all__ = [
    "VoiceState",
    "VoiceEvent",
    "VoiceUIState",
    "VoiceInterface",
]


