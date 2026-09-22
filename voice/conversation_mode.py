"""
RENIX AI - Conversation Mode

Manages continuous, natural voice conversations.

Responsibilities:
- Activate/deactivate conversation mode
- Maintain conversational state
- Track turns
- Handle user interruptions
- Detect end-of-turn conditions
- Support continuous listening
- Coordinate STT/TTS state
- Prevent RENIX from responding to its own voice
- Handle follow-up questions
- Manage conversation timeouts
- Provide clean lifecycle controls

This module does NOT directly implement:
- Microphone hardware
- Speech recognition
- Text-to-speech synthesis
- Wake-word detection
- Clap detection

Those responsibilities belong to the corresponding voice modules.
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
    "RENIX.voice.conversation_mode"
)


# ============================================================================
# ENUMS
# ============================================================================


class ConversationState(str, Enum):
    """Current RENIX conversation state."""

    INACTIVE = "inactive"

    ACTIVATING = "activating"

    LISTENING = "listening"

    PROCESSING = "processing"

    THINKING = "thinking"

    SPEAKING = "speaking"

    INTERRUPTED = "interrupted"

    PAUSED = "paused"

    ENDING = "ending"

    ERROR = "error"


class ConversationEndReason(str, Enum):
    """Why a conversation ended."""

    USER_REQUEST = "user_request"

    TIMEOUT = "timeout"

    SYSTEM_REQUEST = "system_request"

    ERROR = "error"

    SHUTDOWN = "shutdown"

    WAKE_WORD = "wake_word"

    INACTIVITY = "inactivity"

    UNKNOWN = "unknown"


class TurnType(str, Enum):
    """Conversation turn type."""

    USER = "user"

    RENIX = "renix"

    SYSTEM = "system"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class ConversationConfig:
    """Configuration for continuous conversation."""

    enabled: bool = True

    continuous_mode: bool = True

    max_duration_seconds: float = 1800.0

    inactivity_timeout_seconds: float = 20.0

    follow_up_timeout_seconds: float = 8.0

    response_timeout_seconds: float = 60.0

    interruption_enabled: bool = True

    barge_in_enabled: bool = True

    allow_follow_up: bool = True

    auto_resume_listening: bool = True

    suppress_self_audio: bool = True

    require_wake_word_for_new_session: bool = True

    allow_wake_word_during_session: bool = True

    max_turns: int = 200

    min_user_text_length: int = 1

    silence_ends_turn: bool = True

    emit_partial_events: bool = True


@dataclass
class ConversationTurn:
    """Represents one conversational turn."""

    turn_id: str

    conversation_id: str

    turn_type: TurnType

    text: str

    timestamp: float

    duration_seconds: float = 0.0

    interrupted: bool = False

    completed: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "turn_type": self.turn_type.value,
            "text": self.text,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "interrupted": self.interrupted,
            "completed": self.completed,
            "metadata": self.metadata,
        }


@dataclass
class ConversationSession:
    """Complete conversation session."""

    conversation_id: str

    started_at: float

    state: ConversationState = (
        ConversationState.INACTIVE
    )

    ended_at: Optional[float] = None

    end_reason: Optional[
        ConversationEndReason
    ] = None

    turns: list[
        ConversationTurn
    ] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    last_user_activity: float = field(
        default_factory=time.monotonic
    )

    last_renix_activity: float = field(
        default_factory=time.monotonic
    )

    interruption_count: int = 0

    user_turn_count: int = 0

    renix_turn_count: int = 0

    def duration(
        self,
    ) -> float:

        end = (
            self.ended_at
            if self.ended_at is not None
            else time.time()
        )

        return max(
            0.0,
            end - self.started_at,
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "conversation_id": self.conversation_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration": self.duration(),
            "state": self.state.value,
            "end_reason": (
                self.end_reason.value
                if self.end_reason
                else None
            ),
            "turn_count": len(
                self.turns
            ),
            "user_turn_count": (
                self.user_turn_count
            ),
            "renix_turn_count": (
                self.renix_turn_count
            ),
            "interruption_count": (
                self.interruption_count
            ),
            "metadata": self.metadata,
        }


@dataclass
class ConversationEvent:
    """Event emitted by the conversation manager."""

    event_type: str

    conversation_id: Optional[str]

    timestamp: float

    data: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# EXCEPTIONS
# ============================================================================


class ConversationModeError(
    RuntimeError
):
    """Base conversation-mode exception."""


class ConversationNotActiveError(
    ConversationModeError
):
    """Raised when an operation requires an active session."""


class ConversationAlreadyActiveError(
    ConversationModeError
):
    """Raised when attempting to start an active session."""


# ============================================================================
# CONVERSATION MANAGER
# ============================================================================


class ConversationMode:
    """
    Main RENIX continuous conversation manager.

    The manager acts as the bridge between:

        Wake Word
             ↓
        Conversation Mode
             ↓
        Microphone / STT
             ↓
        RENIX Core
             ↓
        TTS
             ↓
        Conversation Mode
             ↓
        Listening again
    """

    def __init__(
        self,
        config: Optional[
            ConversationConfig
        ] = None,
        *,
        on_state_change: Optional[
            Callable[
                [ConversationState],
                None,
            ]
        ] = None,
        on_event: Optional[
            Callable[
                [ConversationEvent],
                None,
            ]
        ] = None,
        on_user_text: Optional[
            Callable[
                [str],
                Any,
            ]
        ] = None,
        on_response: Optional[
            Callable[
                [str],
                Any,
            ]
        ] = None,
        on_start_listening: Optional[
            Callable[[],
                    Any]
        ] = None,
        on_stop_listening: Optional[
            Callable[[],
                    Any]
        ] = None,
        on_start_speaking: Optional[
            Callable[[],
                    Any]
        ] = None,
        on_stop_speaking: Optional[
            Callable[[],
                    Any]
        ] = None,
    ) -> None:

        self.config = (
            config
            or ConversationConfig()
        )

        self._lock = threading.RLock()

        self._state = (
            ConversationState.INACTIVE
        )

        self._session: Optional[
            ConversationSession
        ] = None

        self._current_user_turn: Optional[
            ConversationTurn
        ] = None

        self._current_renix_turn: Optional[
            ConversationTurn
        ] = None

        self._callbacks: dict[
            str,
            list[Callable[..., Any]],
        ] = {}

        self._on_state_change = (
            on_state_change
        )

        self._on_event = on_event

        self._on_user_text = (
            on_user_text
        )

        self._on_response = (
            on_response
        )

        self._on_start_listening = (
            on_start_listening
        )

        self._on_stop_listening = (
            on_stop_listening
        )

        self._on_start_speaking = (
            on_start_speaking
        )

        self._on_stop_speaking = (
            on_stop_speaking
        )

        self._shutdown_event = (
            threading.Event()
        )

        self._monitor_thread: Optional[
            threading.Thread
        ] = None

        self._monitor_running = False

        self._speech_output_active = False

        self._interrupted = False

        self._last_activity = (
            time.monotonic()
        )

        self._session_started_monotonic: Optional[
            float
        ] = None

        self._validate_config()

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    def _validate_config(
        self,
    ) -> None:

        if (
            self.config.max_duration_seconds
            <= 0
        ):

            raise ValueError(
                "max_duration_seconds must be positive."
            )

        if (
            self.config.inactivity_timeout_seconds
            <= 0
        ):

            raise ValueError(
                "inactivity_timeout_seconds must be positive."
            )

        if (
            self.config.follow_up_timeout_seconds
            <= 0
        ):

            raise ValueError(
                "follow_up_timeout_seconds must be positive."
            )

        if self.config.max_turns <= 0:

            raise ValueError(
                "max_turns must be positive."
            )

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def state(
        self,
    ) -> ConversationState:

        with self._lock:

            return self._state

    @property
    def active(
        self,
    ) -> bool:

        return self.state not in (
            ConversationState.INACTIVE,
            ConversationState.ENDING,
            ConversationState.ERROR,
        )

    @property
    def conversation_id(
        self,
    ) -> Optional[str]:

        with self._lock:

            if self._session is None:

                return None

            return (
                self._session.conversation_id
            )

    @property
    def session(
        self,
    ) -> Optional[
        ConversationSession
    ]:

        with self._lock:

            return self._session

    @property
    def listening(
        self,
    ) -> bool:

        return (
            self.state
            == ConversationState.LISTENING
        )

    @property
    def speaking(
        self,
    ) -> bool:

        with self._lock:

            return self._speech_output_active

    # ========================================================================
    # EVENT SYSTEM
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

    def _emit_event(
        self,
        event_type: str,
        data: Optional[
            dict[str, Any]
        ] = None,
    ) -> None:

        event = ConversationEvent(
            event_type=event_type,
            conversation_id=(
                self.conversation_id
            ),
            timestamp=time.time(),
            data=data or {},
        )

        callback = self._on_event

        if callback is not None:

            try:

                callback(event)

            except Exception:

                logger.exception(
                    "Conversation event callback failed."
                )

        with self._lock:

            callbacks = list(
                self._callbacks.get(
                    event_type,
                    [],
                )
            )

        for handler in callbacks:

            try:

                handler(event)

            except Exception:

                logger.exception(
                    "Conversation callback failed: %s",
                    event_type,
                )

    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================

    def _set_state(
        self,
        new_state: ConversationState,
    ) -> None:

        with self._lock:

            old_state = self._state

            if old_state == new_state:

                return

            self._state = new_state

            if self._session is not None:

                self._session.state = (
                    new_state
                )

        logger.debug(
            "Conversation state: %s -> %s",
            old_state.value,
            new_state.value,
        )

        callback = self._on_state_change

        if callback is not None:

            try:

                callback(
                    new_state
                )

            except Exception:

                logger.exception(
                    "State-change callback failed."
                )

        self._emit_event(
            "state_changed",
            {
                "old_state": old_state.value,
                "new_state": new_state.value,
            },
        )

    # ========================================================================
    # SESSION LIFECYCLE
    # ========================================================================

    def start(
        self,
        *,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> str:

        with self._lock:

            if self.active:

                raise ConversationAlreadyActiveError(
                    "A conversation is already active."
                )

            if not self.config.enabled:

                raise ConversationModeError(
                    "Conversation mode is disabled."
                )

            conversation_id = str(
                uuid.uuid4()
            )

            now = time.time()

            self._session = ConversationSession(
                conversation_id=conversation_id,
                started_at=now,
                state=ConversationState.ACTIVATING,
                metadata=dict(
                    metadata or {}
                ),
                last_user_activity=time.monotonic(),
                last_renix_activity=time.monotonic(),
            )

            self._session_started_monotonic = (
                time.monotonic()
            )

            self._last_activity = (
                time.monotonic()
            )

            self._interrupted = False

            self._speech_output_active = False

        self._set_state(
            ConversationState.ACTIVATING
        )

        self._emit_event(
            "conversation_started",
            {
                "conversation_id": (
                    conversation_id
                ),
            },
        )

        self._start_monitor()

        self._begin_listening()

        return conversation_id

    def end(
        self,
        reason: ConversationEndReason = (
            ConversationEndReason.USER_REQUEST
        ),
    ) -> None:

        with self._lock:

            if not self.active:

                return

            session = self._session

            if session is None:

                return

            session.end_reason = reason

        self._set_state(
            ConversationState.ENDING
        )

        self._stop_listening()

        if self._speech_output_active:

            self.stop_speaking()

        with self._lock:

            if self._session is not None:

                self._session.ended_at = (
                    time.time()
                )

                self._session.state = (
                    ConversationState.INACTIVE
                )

            conversation_id = (
                self._session.conversation_id
                if self._session
                else None
            )

            session_snapshot = (
                self._session.to_dict()
                if self._session
                else {}
            )

            self._session = None

            self._current_user_turn = None

            self._current_renix_turn = None

            self._session_started_monotonic = None

            self._interrupted = False

            self._speech_output_active = False

        self._set_state(
            ConversationState.INACTIVE
        )

        self._emit_event(
            "conversation_ended",
            {
                "conversation_id": conversation_id,
                "reason": reason.value,
                "session": session_snapshot,
            },
        )

    def pause(
        self,
    ) -> None:

        if not self.active:

            return

        self._stop_listening()

        self._set_state(
            ConversationState.PAUSED
        )

        self._emit_event(
            "conversation_paused"
        )

    def resume(
        self,
    ) -> None:

        if not self.active:

            raise ConversationNotActiveError(
                "No active conversation."
            )

        self._last_activity = (
            time.monotonic()
        )

        self._set_state(
            ConversationState.LISTENING
        )

        self._begin_listening()

        self._emit_event(
            "conversation_resumed"
        )

    # ========================================================================
    # LISTENING
    # ========================================================================

    def _begin_listening(
        self,
    ) -> None:

        if not self.active:

            return

        if self._speech_output_active:

            return

        self._set_state(
            ConversationState.LISTENING
        )

        callback = (
            self._on_start_listening
        )

        if callback is not None:

            try:

                callback()

            except Exception:

                logger.exception(
                    "Failed to start listening."
                )

        self._emit_event(
            "listening_started"
        )

    def _stop_listening(
        self,
    ) -> None:

        callback = (
            self._on_stop_listening
        )

        if callback is not None:

            try:

                callback()

            except Exception:

                logger.exception(
                    "Failed to stop listening."
                )

        self._emit_event(
            "listening_stopped"
        )

    # ========================================================================
    # USER INPUT
    # ========================================================================

    def receive_user_text(
        self,
        text: str,
        *,
        final: bool = True,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Optional[
        ConversationTurn
    ]:

        if not self.active:

            raise ConversationNotActiveError(
                "Cannot receive user text "
                "without an active conversation."
            )

        text = (
            str(text)
            .strip()
        )

        if not text:

            return None

        if (
            len(text)
            < self.config.min_user_text_length
        ):

            return None

        self._last_activity = (
            time.monotonic()
        )

        with self._lock:

            if self._session is None:

                return None

            self._session.last_user_activity = (
                time.monotonic()
            )

        if not final:

            self._emit_event(
                "user_partial_text",
                {
                    "text": text,
                },
            )

            return None

        self._set_state(
            ConversationState.PROCESSING
        )

        turn = ConversationTurn(
            turn_id=str(
                uuid.uuid4()
            ),
            conversation_id=(
                self.conversation_id
                or ""
            ),
            turn_type=TurnType.USER,
            text=text,
            timestamp=time.time(),
            metadata=dict(
                metadata or {}
            ),
        )

        with self._lock:

            if self._session is None:

                return None

            if (
                len(
                    self._session.turns
                )
                >= self.config.max_turns
            ):

                logger.warning(
                    "Maximum conversation turns reached."
                )

                self.end(
                    ConversationEndReason.INACTIVITY
                )

                return None

            self._session.turns.append(
                turn
            )

            self._session.user_turn_count += 1

            self._session.last_user_activity = (
                time.monotonic()
            )

            self._current_user_turn = (
                turn
            )

        self._emit_event(
            "user_turn",
            {
                "turn": turn.to_dict(),
            },
        )

        callback = self._on_user_text

        if callback is not None:

            try:

                callback(text)

            except Exception:

                logger.exception(
                    "User-text callback failed."
                )

        return turn

    # ========================================================================
    # RENIX RESPONSE
    # ========================================================================

    def begin_response(
        self,
    ) -> None:

        if not self.active:

            raise ConversationNotActiveError(
                "No active conversation."
            )

        self._set_state(
            ConversationState.THINKING
        )

        self._emit_event(
            "response_processing_started"
        )

    def begin_speaking(
        self,
        *,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> str:

        if not self.active:

            raise ConversationNotActiveError(
                "No active conversation."
            )

        self._stop_listening()

        self._speech_output_active = True

        self._interrupted = False

        self._set_state(
            ConversationState.SPEAKING
        )

        turn = ConversationTurn(
            turn_id=str(
                uuid.uuid4()
            ),
            conversation_id=(
                self.conversation_id
                or ""
            ),
            turn_type=TurnType.RENIX,
            text="",
            timestamp=time.time(),
            completed=False,
            metadata=dict(
                metadata or {}
            ),
        )

        with self._lock:

            if self._session is not None:

                self._session.turns.append(
                    turn
                )

                self._session.renix_turn_count += 1

                self._session.last_renix_activity = (
                    time.monotonic()
                )

            self._current_renix_turn = (
                turn
            )

        callback = (
            self._on_start_speaking
        )

        if callback is not None:

            try:

                callback()

            except Exception:

                logger.exception(
                    "Start-speaking callback failed."
                )

        self._emit_event(
            "speaking_started"
        )

        return turn.turn_id

    def receive_response_text(
        self,
        text: str,
        *,
        final: bool = False,
    ) -> None:

        if not self.active:

            return

        text = str(
            text or ""
        )

        with self._lock:

            turn = (
                self._current_renix_turn
            )

            if turn is None:

                return

            turn.text = text

            if final:

                turn.completed = True

            if self._session is not None:

                self._session.last_renix_activity = (
                    time.monotonic()
                )

        self._emit_event(
            "renix_response_text",
            {
                "text": text,
                "final": final,
            },
        )

    def finish_response(
        self,
        text: Optional[str] = None,
    ) -> None:

        if not self.active:

            return

        if text is not None:

            self.receive_response_text(
                text,
                final=True,
            )

        with self._lock:

            turn = (
                self._current_renix_turn
            )

            if turn is not None:

                turn.completed = True

                turn.duration_seconds = (
                    max(
                        0.0,
                        time.time()
                        - turn.timestamp,
                    )
                )

            self._current_renix_turn = None

            self._speech_output_active = False

            self._last_activity = (
                time.monotonic()
            )

        callback = (
            self._on_stop_speaking
        )

        if callback is not None:

            try:

                callback()

            except Exception:

                logger.exception(
                    "Stop-speaking callback failed."
                )

        self._emit_event(
            "speaking_finished"
        )

        if self.active:

            if (
                self.config.auto_resume_listening
            ):

                self._begin_listening()

    # ========================================================================
    # RESPONSE SHORTCUT
    # ========================================================================

    def respond(
        self,
        text: str,
        *,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> None:

        if not self.active:

            raise ConversationNotActiveError(
                "No active conversation."
            )

        self.begin_response()

        self.begin_speaking(
            metadata=metadata
        )

        self.receive_response_text(
            text,
            final=True,
        )

        callback = self._on_response

        if callback is not None:

            try:

                callback(text)

            except Exception:

                logger.exception(
                    "Response callback failed."
                )

        self.finish_response()

    # ========================================================================
    # INTERRUPTION / BARGE-IN
    # ========================================================================

    def interrupt(
        self,
        *,
        reason: str = "user_interruption",
    ) -> bool:

        if not self.active:

            return False

        if not self.config.interruption_enabled:

            return False

        with self._lock:

            if not self._speech_output_active:

                return False

            self._interrupted = True

            self._speech_output_active = False

            if self._session is not None:

                self._session.interruption_count += 1

            turn = (
                self._current_renix_turn
            )

            if turn is not None:

                turn.interrupted = True

                turn.completed = False

        self._set_state(
            ConversationState.INTERRUPTED
        )

        callback = (
            self._on_stop_speaking
        )

        if callback is not None:

            try:

                callback()

            except Exception:

                logger.exception(
                    "Failed to stop speech after interruption."
                )

        self._emit_event(
            "conversation_interrupted",
            {
                "reason": reason,
            },
        )

        if self.active:

            self._begin_listening()

        return True

    def handle_barge_in(
        self,
        text: Optional[str] = None,
    ) -> bool:

        if not self.config.barge_in_enabled:

            return False

        interrupted = self.interrupt(
            reason="barge_in"
        )

        if (
            interrupted
            and text
        ):

            self.receive_user_text(
                text
            )

        return interrupted

    # ========================================================================
    # SPEECH OUTPUT PROTECTION
    # ========================================================================

    def set_speech_output_active(
        self,
        active: bool,
    ) -> None:

        with self._lock:

            self._speech_output_active = (
                bool(active)
            )

        self._emit_event(
            "speech_output_state_changed",
            {
                "active": bool(active),
            },
        )

    def should_ignore_microphone_input(
        self,
    ) -> bool:

        if not self.config.suppress_self_audio:

            return False

        with self._lock:

            return (
                self._speech_output_active
                and not self._interrupted
            )

    # ========================================================================
    # FOLLOW-UP HANDLING
    # ========================================================================

    def should_keep_conversation_alive(
        self,
    ) -> bool:

        if not self.active:

            return False

        if not self.config.continuous_mode:

            return False

        if (
            self.config.max_duration_seconds
            > 0
        ):

            if (
                self._session_started_monotonic
                is not None
            ):

                elapsed = (
                    time.monotonic()
                    - self._session_started_monotonic
                )

                if (
                    elapsed
                    >= self.config.max_duration_seconds
                ):

                    return False

        return True

    def handle_silence(
        self,
    ) -> None:

        if not self.active:

            return

        if not self.config.silence_ends_turn:

            return

        now = time.monotonic()

        with self._lock:

            elapsed = (
                now
                - self._last_activity
            )

        timeout = (
            self.config.follow_up_timeout_seconds
            if self._session
            and self._session.renix_turn_count
            > 0
            else self.config.inactivity_timeout_seconds
        )

        if elapsed >= timeout:

            if self.config.allow_follow_up:

                self._emit_event(
                    "follow_up_timeout",
                    {
                        "elapsed": elapsed,
                    },
                )

                if (
                    self.config.continuous_mode
                    and self._session
                    and self._session.user_turn_count
                    > 0
                ):

                    self.end(
                        ConversationEndReason.INACTIVITY
                    )

            else:

                self.end(
                    ConversationEndReason.TIMEOUT
                )

    # ========================================================================
    # WAKE WORD
    # ========================================================================

    def handle_wake_word(
        self,
        wake_word: str,
    ) -> None:

        wake_word = (
            str(wake_word)
            .strip()
        )

        if not wake_word:

            return

        if not self.active:

            self.start(
                metadata={
                    "activation": "wake_word",
                    "wake_word": wake_word,
                }
            )

            self._emit_event(
                "wake_word_activated",
                {
                    "wake_word": wake_word,
                },
            )

            return

        if (
            self.config.allow_wake_word_during_session
        ):

            self._last_activity = (
                time.monotonic()
            )

            self._emit_event(
                "wake_word_detected",
                {
                    "wake_word": wake_word,
                },
            )

    # ========================================================================
    # MONITORING
    # ========================================================================

    def _start_monitor(
        self,
    ) -> None:

        with self._lock:

            if self._monitor_running:

                return

            self._monitor_running = True

            self._shutdown_event.clear()

        self._monitor_thread = (
            threading.Thread(
                target=self._monitor_loop,
                name="RENIX-ConversationMonitor",
                daemon=True,
            )
        )

        self._monitor_thread.start()

    def _stop_monitor(
        self,
    ) -> None:

        self._shutdown_event.set()

        with self._lock:

            self._monitor_running = False

        thread = self._monitor_thread

        if (
            thread is not None
            and thread is not threading.current_thread()
        ):

            thread.join(
                timeout=2.0
            )

        self._monitor_thread = None

    def _monitor_loop(
        self,
    ) -> None:

        while not self._shutdown_event.wait(
            0.5
        ):

            try:

                if not self.active:

                    continue

                if (
                    self._session_started_monotonic
                    is not None
                ):

                    elapsed = (
                        time.monotonic()
                        - self._session_started_monotonic
                    )

                    if (
                        elapsed
                        >= self.config.max_duration_seconds
                    ):

                        self.end(
                            ConversationEndReason.TIMEOUT
                        )

                        continue

                if (
                    self.state
                    == ConversationState.LISTENING
                ):

                    self.handle_silence()

            except Exception:

                logger.exception(
                    "Conversation monitor error."
                )

    # ========================================================================
    # TURN HELPERS
    # ========================================================================

    def get_turns(
        self,
        *,
        turn_type: Optional[
            TurnType
        ] = None,
    ) -> list[ConversationTurn]:

        with self._lock:

            if self._session is None:

                return []

            turns = list(
                self._session.turns
            )

        if turn_type is None:

            return turns

        return [
            turn
            for turn in turns
            if turn.turn_type
            == turn_type
        ]

    def get_last_user_text(
        self,
    ) -> Optional[str]:

        turns = self.get_turns(
            turn_type=TurnType.USER
        )

        if not turns:

            return None

        return turns[-1].text

    def get_last_response(
        self,
    ) -> Optional[str]:

        turns = self.get_turns(
            turn_type=TurnType.RENIX
        )

        if not turns:

            return None

        return turns[-1].text

    def get_recent_context(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:

        if limit <= 0:

            return []

        turns = self.get_turns()

        return [
            turn.to_dict()
            for turn in turns[-limit:]
        ]

    # ========================================================================
    # ACTIVITY
    # ========================================================================

    def mark_activity(
        self,
    ) -> None:

        self._last_activity = (
            time.monotonic()
        )

        with self._lock:

            if self._session is not None:

                self._session.last_user_activity = (
                    time.monotonic()
                )

    def seconds_since_activity(
        self,
    ) -> float:

        return max(
            0.0,
            time.monotonic()
            - self._last_activity,
        )

    # ========================================================================
    # STATUS
    # ========================================================================

    def get_status(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            session = self._session

            return {
                "enabled": (
                    self.config.enabled
                ),
                "active": self.active,
                "state": self._state.value,
                "conversation_id": (
                    session.conversation_id
                    if session
                    else None
                ),
                "turn_count": (
                    len(session.turns)
                    if session
                    else 0
                ),
                "user_turn_count": (
                    session.user_turn_count
                    if session
                    else 0
                ),
                "renix_turn_count": (
                    session.renix_turn_count
                    if session
                    else 0
                ),
                "interruption_count": (
                    session.interruption_count
                    if session
                    else 0
                ),
                "speaking": (
                    self._speech_output_active
                ),
                "interrupted": (
                    self._interrupted
                ),
                "seconds_since_activity": (
                    self.seconds_since_activity()
                ),
            }

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    def shutdown(
        self,
    ) -> None:

        try:

            if self.active:

                self.end(
                    ConversationEndReason.SHUTDOWN
                )

        finally:

            self._stop_monitor()

            with self._lock:

                self._state = (
                    ConversationState.INACTIVE
                )

                self._session = None

                self._current_user_turn = None

                self._current_renix_turn = None

                self._speech_output_active = False

                self._interrupted = False


# ============================================================================
# DEFAULT INSTANCE
# ============================================================================


_default_conversation_mode: Optional[
    ConversationMode
] = None

_default_lock = threading.RLock()


def get_conversation_mode() -> ConversationMode:
    """Return the process-wide RENIX conversation manager."""

    global _default_conversation_mode

    with _default_lock:

        if _default_conversation_mode is None:

            _default_conversation_mode = (
                ConversationMode()
            )

        return _default_conversation_mode


def start_conversation(
    *,
    metadata: Optional[
        dict[str, Any]
    ] = None,
) -> str:

    return get_conversation_mode().start(
        metadata=metadata
    )


def end_conversation(
    reason: ConversationEndReason = (
        ConversationEndReason.USER_REQUEST
    ),
) -> None:

    get_conversation_mode().end(
        reason
    )


def is_conversation_active() -> bool:

    return get_conversation_mode().active


def get_conversation_status() -> dict[str, Any]:

    return (
        get_conversation_mode()
        .get_status()
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "ConversationState",
    "ConversationEndReason",
    "TurnType",
    "ConversationConfig",
    "ConversationTurn",
    "ConversationSession",
    "ConversationEvent",
    "ConversationModeError",
    "ConversationNotActiveError",
    "ConversationAlreadyActiveError",
    "ConversationMode",
    "get_conversation_mode",
    "start_conversation",
    "end_conversation",
    "is_conversation_active",
    "get_conversation_status",
]


