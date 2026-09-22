"""
RENIX Holographic UI
Command UI
=================

Command interface for the RENIX holographic HUD.

Responsibilities:
- Display current voice/gesture command
- Show command history
- Show command execution state
- Show AI response
- Provide command suggestions
- Provide renderer-independent UI snapshots
- Support callbacks from the main UI/orchestrator
"""

from __future__ import annotations

import threading
import time
import uuid

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ============================================================================
# ENUMS
# ============================================================================


class CommandState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    RECEIVED = "received"
    PROCESSING = "processing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CommandSource(str, Enum):
    VOICE = "voice"
    GESTURE = "gesture"
    KEYBOARD = "keyboard"
    SYSTEM = "system"
    AI = "ai"
    UNKNOWN = "unknown"


class CommandPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# COMMAND
# ============================================================================


@dataclass
class Command:
    id: str

    text: str

    source: CommandSource = CommandSource.UNKNOWN

    state: CommandState = CommandState.RECEIVED

    priority: CommandPriority = (
        CommandPriority.NORMAL
    )

    created_at: float = field(
        default_factory=time.time
    )

    started_at: Optional[float] = None

    completed_at: Optional[float] = None

    response: Optional[str] = None

    error: Optional[str] = None

    progress: Optional[float] = None

    intent: Optional[str] = None

    confidence: Optional[float] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def start(self) -> None:

        self.state = CommandState.PROCESSING

        if self.started_at is None:
            self.started_at = time.time()

    def execute(self) -> None:

        self.state = CommandState.EXECUTING

        if self.started_at is None:
            self.started_at = time.time()

    def complete(
        self,
        response: Optional[str] = None,
    ) -> None:

        self.state = CommandState.COMPLETED

        self.completed_at = time.time()

        self.progress = 1.0

        if response is not None:
            self.response = response

    def fail(
        self,
        error: str,
    ) -> None:

        self.state = CommandState.FAILED

        self.completed_at = time.time()

        self.error = error

    def cancel(self) -> None:

        self.state = CommandState.CANCELLED

        self.completed_at = time.time()

    @property
    def duration(self) -> Optional[float]:

        if self.started_at is None:
            return None

        end = (
            self.completed_at
            or time.time()
        )

        return max(
            0.0,
            end - self.started_at,
        )


# ============================================================================
# SUGGESTION
# ============================================================================


@dataclass
class CommandSuggestion:

    id: str

    text: str

    description: Optional[str] = None

    icon: Optional[str] = None

    confidence: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# COMMAND UI
# ============================================================================


class CommandUI:

    def __init__(
        self,
        *,
        max_history: int = 50,
        max_suggestions: int = 6,
    ) -> None:

        self.max_history = max(
            1,
            int(max_history),
        )

        self.max_suggestions = max(
            1,
            int(max_suggestions),
        )

        self._commands: dict[
            str,
            Command,
        ] = {}

        self._history: list[str] = []

        self._active_command_id: Optional[
            str
        ] = None

        self._suggestions: list[
            CommandSuggestion
        ] = []

        self._input_text = ""

        self._is_focused = False

        self._is_listening = False

        self._running = False

        self._lock = threading.RLock()

        self._callbacks: dict[
            str,
            list[Callable[..., None]],
        ] = {}

        self._last_update = time.monotonic()

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self) -> None:

        with self._lock:

            if self._running:
                return

            self._running = True

            self._last_update = (
                time.monotonic()
            )

        self._emit(
            "started",
            self,
        )

    def stop(self) -> None:

        with self._lock:

            self._running = False

        self._emit(
            "stopped",
            self,
        )

    @property
    def running(self) -> bool:

        with self._lock:

            return self._running

    def update(
        self,
        delta_time: Optional[float] = None,
    ) -> None:

        now = time.monotonic()

        if delta_time is None:

            delta_time = (
                now - self._last_update
            )

        self._last_update = now

        self._emit(
            "updated",
            delta_time,
        )

    # ========================================================================
    # INPUT
    # ========================================================================

    def set_input(
        self,
        text: str,
    ) -> None:

        with self._lock:

            self._input_text = (
                text or ""
            )

        self._emit(
            "input_changed",
            self._input_text,
        )

    def append_input(
        self,
        text: str,
    ) -> None:

        if not text:
            return

        with self._lock:

            self._input_text += text

        self._emit(
            "input_changed",
            self._input_text,
        )

    def clear_input(self) -> None:

        with self._lock:

            self._input_text = ""

        self._emit(
            "input_changed",
            "",
        )

    def get_input(self) -> str:

        with self._lock:

            return self._input_text

    # ========================================================================
    # FOCUS
    # ========================================================================

    def focus(self) -> None:

        with self._lock:

            self._is_focused = True

        self._emit(
            "focus_changed",
            True,
        )

    def unfocus(self) -> None:

        with self._lock:

            self._is_focused = False

        self._emit(
            "focus_changed",
            False,
        )

    @property
    def focused(self) -> bool:

        with self._lock:

            return self._is_focused

    # ========================================================================
    # LISTENING
    # ========================================================================

    def start_listening(self) -> None:

        with self._lock:

            self._is_listening = True

        self._emit(
            "listening_started",
        )

    def stop_listening(self) -> None:

        with self._lock:

            self._is_listening = False

        self._emit(
            "listening_stopped",
        )

    @property
    def listening(self) -> bool:

        with self._lock:

            return self._is_listening

    # ========================================================================
    # COMMAND CREATION
    # ========================================================================

    def submit(
        self,
        text: Optional[str] = None,
        *,
        source: CommandSource = (
            CommandSource.UNKNOWN
        ),
        priority: CommandPriority = (
            CommandPriority.NORMAL
        ),
        intent: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Optional[str]:

        if text is None:

            text = self.get_input()

        text = text.strip()

        if not text:
            return None

        command = Command(
            id=uuid.uuid4().hex,
            text=text,
            source=source,
            state=CommandState.RECEIVED,
            priority=priority,
            intent=intent,
            confidence=(
                self._normalize_confidence(
                    confidence
                )
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        with self._lock:

            self._commands[
                command.id
            ] = command

            self._history.append(
                command.id
            )

            self._active_command_id = (
                command.id
            )

            self._trim_history()

            self._input_text = ""

        self._emit(
            "command_submitted",
            command,
        )

        return command.id

    # ========================================================================
    # COMMAND STATE
    # ========================================================================

    def set_state(
        self,
        command_id: str,
        state: CommandState,
    ) -> bool:

        command = self.get_command(
            command_id
        )

        if command is None:
            return False

        command.state = state

        if state in (
            CommandState.PROCESSING,
            CommandState.EXECUTING,
        ):

            if command.started_at is None:
                command.started_at = time.time()

        if state in (
            CommandState.COMPLETED,
            CommandState.FAILED,
            CommandState.CANCELLED,
        ):

            command.completed_at = (
                time.time()
            )

        self._emit(
            "command_state_changed",
            command,
        )

        return True

    def begin_processing(
        self,
        command_id: str,
    ) -> bool:

        command = self.get_command(
            command_id
        )

        if command is None:
            return False

        command.start()

        self._emit(
            "command_processing",
            command,
        )

        return True

    def begin_execution(
        self,
        command_id: str,
    ) -> bool:

        command = self.get_command(
            command_id
        )

        if command is None:
            return False

        command.execute()

        self._emit(
            "command_execution",
            command,
        )

        return True

    def complete(
        self,
        command_id: str,
        response: Optional[str] = None,
    ) -> bool:

        command = self.get_command(
            command_id
        )

        if command is None:
            return False

        command.complete(
            response
        )

        self._emit(
            "command_completed",
            command,
        )

        return True

    def fail(
        self,
        command_id: str,
        error: str,
    ) -> bool:

        command = self.get_command(
            command_id
        )

        if command is None:
            return False

        command.fail(
            error
        )

        self._emit(
            "command_failed",
            command,
        )

        return True

    def cancel(
        self,
        command_id: str,
    ) -> bool:

        command = self.get_command(
            command_id
        )

        if command is None:
            return False

        command.cancel()

        self._emit(
            "command_cancelled",
            command,
        )

        return True

    # ========================================================================
    # PROGRESS
    # ========================================================================

    def set_progress(
        self,
        command_id: str,
        progress: Optional[float],
    ) -> bool:

        command = self.get_command(
            command_id
        )

        if command is None:
            return False

        command.progress = (
            self._normalize_progress(
                progress
            )
        )

        self._emit(
            "command_progress",
            command,
        )

        return True

    # ========================================================================
    # RESPONSE
    # ========================================================================

    def set_response(
        self,
        command_id: str,
        response: str,
    ) -> bool:

        command = self.get_command(
            command_id
        )

        if command is None:
            return False

        command.response = response

        self._emit(
            "response_changed",
            command,
        )

        return True

    # ========================================================================
    # ACTIVE COMMAND
    # ========================================================================

    def get_active_command(
        self,
    ) -> Optional[Command]:

        with self._lock:

            if (
                self._active_command_id
                is None
            ):
                return None

            return self._commands.get(
                self._active_command_id
            )

    def set_active_command(
        self,
        command_id: Optional[str],
    ) -> bool:

        if command_id is None:

            with self._lock:

                self._active_command_id = None

            self._emit(
                "active_command_changed",
                None,
            )

            return True

        command = self.get_command(
            command_id
        )

        if command is None:
            return False

        with self._lock:

            self._active_command_id = (
                command_id
            )

        self._emit(
            "active_command_changed",
            command,
        )

        return True

    # ========================================================================
    # COMMAND ACCESS
    # ========================================================================

    def get_command(
        self,
        command_id: str,
    ) -> Optional[Command]:

        with self._lock:

            return self._commands.get(
                command_id
            )

    def get_history(
        self,
        limit: Optional[int] = None,
    ) -> list[Command]:

        with self._lock:

            ids = list(
                reversed(
                    self._history
                )
            )

            if limit is not None:

                ids = ids[
                    : max(
                        0,
                        int(limit),
                    )
                ]

            return [
                self._commands[
                    command_id
                ]
                for command_id in ids
                if command_id
                in self._commands
            ]

    # ========================================================================
    # SUGGESTIONS
    # ========================================================================

    def set_suggestions(
        self,
        suggestions: list[
            CommandSuggestion
        ],
    ) -> None:

        cleaned = list(
            suggestions[
                : self.max_suggestions
            ]
        )

        cleaned.sort(
            key=lambda item: item.confidence,
            reverse=True,
        )

        with self._lock:

            self._suggestions = cleaned

        self._emit(
            "suggestions_changed",
            cleaned,
        )

    def add_suggestion(
        self,
        text: str,
        *,
        description: Optional[str] = None,
        icon: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> str:

        suggestion = CommandSuggestion(
            id=uuid.uuid4().hex,
            text=text,
            description=description,
            icon=icon,
            confidence=max(
                0.0,
                min(
                    1.0,
                    float(confidence),
                ),
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        with self._lock:

            self._suggestions.append(
                suggestion
            )

            self._suggestions.sort(
                key=lambda item: item.confidence,
                reverse=True,
            )

            self._suggestions = (
                self._suggestions[
                    : self.max_suggestions
                ]
            )

        self._emit(
            "suggestions_changed",
            self._suggestions,
        )

        return suggestion.id

    def clear_suggestions(self) -> None:

        with self._lock:

            self._suggestions.clear()

        self._emit(
            "suggestions_changed",
            [],
        )

    def get_suggestions(
        self,
    ) -> list[CommandSuggestion]:

        with self._lock:

            return list(
                self._suggestions
            )

    def use_suggestion(
        self,
        suggestion_id: str,
    ) -> Optional[str]:

        suggestion = next(
            (
                item
                for item
                in self._suggestions
                if item.id == suggestion_id
            ),
            None,
        )

        if suggestion is None:
            return None

        self.set_input(
            suggestion.text
        )

        self._emit(
            "suggestion_selected",
            suggestion,
        )

        return suggestion.text

    # ========================================================================
    # SNAPSHOT
    # ========================================================================

    def get_snapshot(
        self,
    ) -> dict[str, Any]:

        active = (
            self.get_active_command()
        )

        return {
            "running": self.running,
            "focused": self.focused,
            "listening": self.listening,
            "input": self.get_input(),
            "active_command": (
                self._serialize_command(
                    active
                )
                if active
                else None
            ),
            "history": [
                self._serialize_command(
                    command
                )
                for command
                in self.get_history(
                    self.max_history
                )
            ],
            "suggestions": [
                self._serialize_suggestion(
                    suggestion
                )
                for suggestion
                in self.get_suggestions()
            ],
        }

    def get_active_snapshot(
        self,
    ) -> Optional[dict[str, Any]]:

        command = (
            self.get_active_command()
        )

        if command is None:
            return None

        return self._serialize_command(
            command
        )

    @staticmethod
    def _serialize_command(
        command: Optional[Command],
    ) -> Optional[dict[str, Any]]:

        if command is None:
            return None

        return {
            "id": command.id,
            "text": command.text,
            "source": command.source.value,
            "state": command.state.value,
            "priority": command.priority.value,
            "created_at": command.created_at,
            "started_at": command.started_at,
            "completed_at": command.completed_at,
            "duration": command.duration,
            "response": command.response,
            "error": command.error,
            "progress": command.progress,
            "intent": command.intent,
            "confidence": command.confidence,
            "metadata": dict(
                command.metadata
            ),
        }

    @staticmethod
    def _serialize_suggestion(
        suggestion: CommandSuggestion,
    ) -> dict[str, Any]:

        return {
            "id": suggestion.id,
            "text": suggestion.text,
            "description": suggestion.description,
            "icon": suggestion.icon,
            "confidence": suggestion.confidence,
            "metadata": dict(
                suggestion.metadata
            ),
        }

    # ========================================================================
    # HISTORY MANAGEMENT
    # ========================================================================

    def _trim_history(self) -> None:

        if (
            len(self._history)
            <= self.max_history
        ):
            return

        remove_count = (
            len(self._history)
            - self.max_history
        )

        old_ids = self._history[
            :remove_count
        ]

        self._history = self._history[
            remove_count:
        ]

        for command_id in old_ids:

            if (
                command_id
                == self._active_command_id
            ):
                continue

            self._commands.pop(
                command_id,
                None,
            )

    def clear_history(self) -> None:

        with self._lock:

            active_id = (
                self._active_command_id
            )

            if active_id is None:

                self._commands.clear()

            else:

                active = self._commands.get(
                    active_id
                )

                self._commands.clear()

                if active is not None:

                    self._commands[
                        active_id
                    ] = active

            self._history.clear()

            if active_id is not None:

                self._history.append(
                    active_id
                )

        self._emit(
            "history_cleared"
        )

    # ========================================================================
    # EVENTS
    # ========================================================================

    def on(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:

        if not event:
            raise ValueError(
                "Event cannot be empty"
            )

        if not callable(callback):
            raise TypeError(
                "Callback must be callable"
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
                # UI callbacks must never
                # crash RENIX.
                pass

    # ========================================================================
    # STATUS
    # ========================================================================

    def status(self) -> dict[str, Any]:

        active = (
            self.get_active_command()
        )

        return {
            "running": self.running,
            "focused": self.focused,
            "listening": self.listening,
            "input_length": len(
                self.get_input()
            ),
            "history_count": len(
                self.get_history()
            ),
            "suggestion_count": len(
                self.get_suggestions()
            ),
            "active_command_id": (
                active.id
                if active
                else None
            ),
            "active_command_state": (
                active.state.value
                if active
                else None
            ),
        }

    # ========================================================================
    # UTILITIES
    # ========================================================================

    @staticmethod
    def _normalize_progress(
        value: Optional[float],
    ) -> Optional[float]:

        if value is None:
            return None

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )

    @staticmethod
    def _normalize_confidence(
        value: Optional[float],
    ) -> Optional[float]:

        if value is None:
            return None

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )


# ============================================================================
# FACTORY
# ============================================================================


def create_command_ui() -> CommandUI:

    ui = CommandUI(
        max_history=50,
        max_suggestions=6,
    )

    ui.start()

    return ui


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "CommandState",
    "CommandSource",
    "CommandPriority",
    "Command",
    "CommandSuggestion",
    "CommandUI",
    "create_command_ui",
]


