"""
RENIX Holographic UI
Terminal UI
==================

UI/state layer for the RENIX terminal interface.

Responsibilities:
- Terminal sessions
- Command input/history
- Output rendering state
- Multiple terminal tabs
- ANSI-style output metadata
- Running/finished/error states
- Search
- Font/opacity settings
- Voice/gesture-friendly actions

Actual command execution belongs in:
    RENIX/coding/terminal_manager.py
    RENIX/computer/computer_controller.py
    RENIX/security/command_safety.py

This file intentionally does NOT execute arbitrary shell commands.
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


class TerminalState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"
    CLOSED = "closed"


class TerminalAction(str, Enum):
    NEW = "new"
    CLOSE = "close"
    CLEAR = "clear"
    RUN = "run"
    STOP = "stop"
    COPY = "copy"
    PASTE = "paste"
    SEARCH = "search"
    NEXT_TAB = "next_tab"
    PREVIOUS_TAB = "previous_tab"
    SPLIT = "split"
    TOGGLE_FULLSCREEN = "toggle_fullscreen"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    RESET_ZOOM = "reset_zoom"


class TerminalViewMode(str, Enum):
    NORMAL = "normal"
    SPLIT = "split"
    FULLSCREEN = "fullscreen"


class TerminalOutputType(str, Enum):
    STDOUT = "stdout"
    STDERR = "stderr"
    SYSTEM = "system"
    COMMAND = "command"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class TerminalLine:

    text: str

    output_type: TerminalOutputType = (
        TerminalOutputType.STDOUT
    )

    timestamp: float = field(
        default_factory=time.time
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class TerminalCommand:

    id: str

    command: str

    timestamp: float = field(
        default_factory=time.time
    )

    exit_code: Optional[int] = None

    duration: Optional[float] = None

    state: TerminalState = (
        TerminalState.IDLE
    )


@dataclass
class TerminalSession:

    id: str

    title: str = "Terminal"

    shell: str = "powershell"

    cwd: str = ""

    state: TerminalState = (
        TerminalState.IDLE
    )

    input_text: str = ""

    lines: list[TerminalLine] = field(
        default_factory=list
    )

    commands: list[TerminalCommand] = field(
        default_factory=list
    )

    history: list[str] = field(
        default_factory=list
    )

    history_index: int = -1

    created_at: float = field(
        default_factory=time.time
    )

    last_updated: float = field(
        default_factory=time.time
    )

    process_id: Optional[int] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class TerminalActionResult:

    action: TerminalAction

    success: bool

    message: str = ""

    data: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# TERMINAL UI
# ============================================================================


class TerminalUI:

    def __init__(
        self,
        *,
        default_shell: str = "powershell",
        max_sessions: int = 20,
        max_history: int = 500,
        max_lines: int = 10000,
    ) -> None:

        self.default_shell = (
            default_shell
        )

        self.max_sessions = max(
            1,
            int(max_sessions),
        )

        self.max_history = max(
            1,
            int(max_history),
        )

        self.max_lines = max(
            100,
            int(max_lines),
        )

        self._sessions: dict[
            str,
            TerminalSession,
        ] = {}

        self._session_order: list[str] = []

        self._active_session_id: Optional[
            str
        ] = None

        self._view_mode = (
            TerminalViewMode.NORMAL
        )

        self._fullscreen = False

        self._search_text = ""

        self._search_matches: list[
            tuple[str, int]
        ] = []

        self._current_search_match = 0

        self._font_size = 14.0

        self._opacity = 0.96

        self._cursor_visible = True

        self._running = False

        self._lock = threading.RLock()

        self._callbacks: dict[
            str,
            list[Callable[..., None]],
        ] = {}

        self._last_update = time.monotonic()

        self._create_initial_session()

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self) -> None:

        with self._lock:

            if self._running:
                return

            self._running = True

        self._emit(
            "started",
            self,
        )

    def stop(self) -> None:

        with self._lock:

            if not self._running:
                return

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
    # INITIAL SESSION
    # ========================================================================

    def _create_initial_session(
        self,
    ) -> None:

        session = TerminalSession(
            id=uuid.uuid4().hex,
            title="Terminal 1",
            shell=self.default_shell,
        )

        self._sessions[
            session.id
        ] = session

        self._session_order.append(
            session.id
        )

        self._active_session_id = (
            session.id
        )

        self._append_line_locked(
            session,
            "RENIX Terminal",
            TerminalOutputType.SYSTEM,
        )

        self._append_line_locked(
            session,
            "Ready.",
            TerminalOutputType.SYSTEM,
        )

    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================

    @property
    def active_session_id(
        self,
    ) -> Optional[str]:

        with self._lock:
            return self._active_session_id

    def get_active_session(
        self,
    ) -> Optional[TerminalSession]:

        with self._lock:

            if (
                self._active_session_id
                is None
            ):
                return None

            return self._sessions.get(
                self._active_session_id
            )

    def get_session(
        self,
        session_id: str,
    ) -> Optional[TerminalSession]:

        with self._lock:
            return self._sessions.get(
                session_id
            )

    def get_sessions(
        self,
    ) -> list[TerminalSession]:

        with self._lock:

            return [
                self._sessions[
                    session_id
                ]
                for session_id
                in self._session_order
                if session_id
                in self._sessions
            ]

    def create_session(
        self,
        *,
        shell: Optional[str] = None,
        cwd: str = "",
        title: Optional[str] = None,
        activate: bool = True,
    ) -> TerminalSession:

        with self._lock:

            if (
                len(self._session_order)
                >= self.max_sessions
            ):

                raise RuntimeError(
                    "Maximum terminal sessions reached."
                )

            session_number = (
                len(self._session_order)
                + 1
            )

            session = TerminalSession(
                id=uuid.uuid4().hex,
                title=(
                    title
                    or f"Terminal {session_number}"
                ),
                shell=(
                    shell
                    or self.default_shell
                ),
                cwd=cwd,
            )

            self._sessions[
                session.id
            ] = session

            self._session_order.append(
                session.id
            )

            if activate:

                self._active_session_id = (
                    session.id
                )

        self._emit(
            "session_created",
            session,
        )

        if activate:

            self._emit(
                "active_session_changed",
                session,
            )

        return session

    def activate_session(
        self,
        session_id: str,
    ) -> bool:

        session = self.get_session(
            session_id
        )

        if session is None:
            return False

        with self._lock:

            self._active_session_id = (
                session_id
            )

        self._emit(
            "active_session_changed",
            session,
        )

        return True

    def close_session(
        self,
        session_id: Optional[str] = None,
    ) -> bool:

        with self._lock:

            target_id = (
                session_id
                or self._active_session_id
            )

            if target_id is None:
                return False

            if target_id not in self._sessions:
                return False

            was_active = (
                target_id
                == self._active_session_id
            )

            self._sessions.pop(
                target_id
            )

            if target_id in self._session_order:

                index = (
                    self._session_order.index(
                        target_id
                    )
                )

                self._session_order.remove(
                    target_id
                )

            else:

                index = 0

            if not self._session_order:

                new_session = self.create_session(
                    activate=True
                )

                self._emit(
                    "session_closed",
                    target_id,
                )

                return True

            if was_active:

                new_index = min(
                    index,
                    len(
                        self._session_order
                    ) - 1,
                )

                new_id = (
                    self._session_order[
                        new_index
                    ]
                )

                self._active_session_id = (
                    new_id
                )

        self._emit(
            "session_closed",
            target_id,
        )

        self._emit(
            "active_session_changed",
            self.get_active_session(),
        )

        return True

    def next_session(
        self,
    ) -> Optional[TerminalSession]:

        with self._lock:

            if not self._session_order:
                return None

            if (
                self._active_session_id
                not in self._session_order
            ):

                target_id = (
                    self._session_order[0]
                )

            else:

                index = (
                    self._session_order.index(
                        self._active_session_id
                    )
                )

                target_id = (
                    self._session_order[
                        (
                            index + 1
                        )
                        % len(
                            self._session_order
                        )
                    ]
                )

        self.activate_session(
            target_id
        )

        return self.get_session(
            target_id
        )

    def previous_session(
        self,
    ) -> Optional[TerminalSession]:

        with self._lock:

            if not self._session_order:
                return None

            if (
                self._active_session_id
                not in self._session_order
            ):

                target_id = (
                    self._session_order[0]
                )

            else:

                index = (
                    self._session_order.index(
                        self._active_session_id
                    )
                )

                target_id = (
                    self._session_order[
                        (
                            index - 1
                        )
                        % len(
                            self._session_order
                        )
                    ]
                )

        self.activate_session(
            target_id
        )

        return self.get_session(
            target_id
        )

    # ========================================================================
    # INPUT
    # ========================================================================

    def set_input(
        self,
        text: str,
    ) -> None:

        session = self.get_active_session()

        if session is None:
            return

        with self._lock:

            session.input_text = (
                text or ""
            )

            session.last_updated = (
                time.time()
            )

        self._emit(
            "input_changed",
            session,
        )

    def get_input(self) -> str:

        session = self.get_active_session()

        if session is None:
            return ""

        with self._lock:
            return session.input_text

    def append_input(
        self,
        text: str,
    ) -> None:

        self.set_input(
            self.get_input() + text
        )

    def clear_input(self) -> None:

        self.set_input("")

    # ========================================================================
    # COMMAND HISTORY
    # ========================================================================

    def add_history(
        self,
        command: str,
        session_id: Optional[str] = None,
    ) -> None:

        command = command.strip()

        if not command:
            return

        session = self.get_session(
            session_id
            or self._active_session_id
            or ""
        )

        if session is None:
            return

        with self._lock:

            if (
                session.history
                and session.history[-1]
                == command
            ):

                session.history_index = (
                    len(session.history)
                )

                return

            session.history.append(
                command
            )

            if (
                len(session.history)
                > self.max_history
            ):

                overflow = (
                    len(session.history)
                    - self.max_history
                )

                session.history = (
                    session.history[
                        overflow:
                    ]
                )

            session.history_index = (
                len(session.history)
            )

    def history_up(
        self,
    ) -> str:

        session = self.get_active_session()

        if session is None:
            return ""

        with self._lock:

            if not session.history:
                return ""

            session.history_index = max(
                0,
                session.history_index - 1,
            )

            value = session.history[
                session.history_index
            ]

            session.input_text = value

        self._emit(
            "input_changed",
            session,
        )

        return value

    def history_down(
        self,
    ) -> str:

        session = self.get_active_session()

        if session is None:
            return ""

        with self._lock:

            if not session.history:
                return ""

            session.history_index = min(
                len(session.history),
                session.history_index + 1,
            )

            if (
                session.history_index
                >= len(session.history)
            ):

                session.input_text = ""

            else:

                session.input_text = (
                    session.history[
                        session.history_index
                    ]
                )

            value = session.input_text

        self._emit(
            "input_changed",
            session,
        )

        return value

    # ========================================================================
    # COMMANDS
    # ========================================================================

    def submit_command(
        self,
        command: Optional[str] = None,
    ) -> Optional[TerminalCommand]:

        session = self.get_active_session()

        if session is None:
            return None

        command_text = (
            command
            if command is not None
            else session.input_text
        ).strip()

        if not command_text:
            return None

        terminal_command = TerminalCommand(
            id=uuid.uuid4().hex,
            command=command_text,
            state=TerminalState.RUNNING,
        )

        with self._lock:

            session.commands.append(
                terminal_command
            )

            self.add_history(
                command_text,
                session.id,
            )

            session.input_text = ""

            session.state = (
                TerminalState.RUNNING
            )

            session.last_updated = (
                time.time()
            )

            self._append_line_locked(
                session,
                f"> {command_text}",
                TerminalOutputType.COMMAND,
            )

        self._emit(
            "command_submitted",
            session,
            terminal_command,
        )

        self._emit(
            "input_changed",
            session,
        )

        return terminal_command

    def command_completed(
        self,
        command_id: str,
        *,
        exit_code: int = 0,
        duration: Optional[float] = None,
    ) -> bool:

        session = self.get_active_session()

        if session is None:
            return False

        with self._lock:

            command = self._find_command_locked(
                session,
                command_id,
            )

            if command is None:
                return False

            command.exit_code = exit_code

            command.duration = duration

            command.state = (
                TerminalState.COMPLETED
                if exit_code == 0
                else TerminalState.ERROR
            )

            session.state = (
                TerminalState.COMPLETED
                if exit_code == 0
                else TerminalState.ERROR
            )

            session.last_updated = (
                time.time()
            )

        self._emit(
            "command_completed",
            session,
            command,
        )

        return True

    def command_stopped(
        self,
        command_id: str,
    ) -> bool:

        session = self.get_active_session()

        if session is None:
            return False

        with self._lock:

            command = self._find_command_locked(
                session,
                command_id,
            )

            if command is None:
                return False

            command.state = (
                TerminalState.STOPPED
            )

            session.state = (
                TerminalState.STOPPED
            )

        self._emit(
            "command_stopped",
            session,
            command,
        )

        return True

    def _find_command_locked(
        self,
        session: TerminalSession,
        command_id: str,
    ) -> Optional[TerminalCommand]:

        for command in reversed(
            session.commands
        ):

            if command.id == command_id:
                return command

        return None

    # ========================================================================
    # OUTPUT
    # ========================================================================

    def write(
        self,
        text: str,
        *,
        output_type: TerminalOutputType = (
            TerminalOutputType.STDOUT
        ),
        session_id: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> None:

        session = self.get_session(
            session_id
            or self._active_session_id
            or ""
        )

        if session is None:
            return

        lines = str(text).splitlines()

        if not lines:

            lines = [""]

        with self._lock:

            for line in lines:

                self._append_line_locked(
                    session,
                    line,
                    output_type,
                    metadata,
                )

            session.last_updated = (
                time.time()
            )

        self._emit(
            "output_added",
            session,
        )

    def write_stdout(
        self,
        text: str,
        **kwargs: Any,
    ) -> None:

        self.write(
            text,
            output_type=(
                TerminalOutputType.STDOUT
            ),
            **kwargs,
        )

    def write_stderr(
        self,
        text: str,
        **kwargs: Any,
    ) -> None:

        self.write(
            text,
            output_type=(
                TerminalOutputType.STDERR
            ),
            **kwargs,
        )

    def write_system(
        self,
        text: str,
        **kwargs: Any,
    ) -> None:

        self.write(
            text,
            output_type=(
                TerminalOutputType.SYSTEM
            ),
            **kwargs,
        )

    def _append_line_locked(
        self,
        session: TerminalSession,
        text: str,
        output_type: TerminalOutputType,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> None:

        session.lines.append(
            TerminalLine(
                text=text,
                output_type=output_type,
                metadata=(
                    dict(metadata)
                    if metadata
                    else {}
                ),
            )
        )

        if (
            len(session.lines)
            > self.max_lines
        ):

            overflow = (
                len(session.lines)
                - self.max_lines
            )

            session.lines = (
                session.lines[
                    overflow:
                ]
            )

    def clear(
        self,
        session_id: Optional[str] = None,
    ) -> None:

        session = self.get_session(
            session_id
            or self._active_session_id
            or ""
        )

        if session is None:
            return

        with self._lock:

            session.lines.clear()

            session.last_updated = (
                time.time()
            )

        self._emit(
            "cleared",
            session,
        )

    # ========================================================================
    # SEARCH
    # ========================================================================

    def search(
        self,
        text: str,
    ) -> int:

        query = text.strip()

        with self._lock:

            self._search_text = query

            self._search_matches.clear()

            self._current_search_match = 0

            if not query:
                return 0

            query_lower = query.lower()

            for session_id in (
                self._session_order
            ):

                session = self._sessions.get(
                    session_id
                )

                if session is None:
                    continue

                for index, line in enumerate(
                    session.lines
                ):

                    if (
                        query_lower
                        in line.text.lower()
                    ):

                        self._search_matches.append(
                            (
                                session_id,
                                index,
                            )
                        )

            count = len(
                self._search_matches
            )

        self._emit(
            "search_changed",
            query,
            count,
        )

        return count

    def next_search_match(
        self,
    ) -> Optional[tuple[str, int]]:

        with self._lock:

            if not self._search_matches:
                return None

            self._current_search_match = (
                (
                    self._current_search_match
                    + 1
                )
                % len(
                    self._search_matches
                )
            )

            result = (
                self._search_matches[
                    self._current_search_match
                ]
            )

        self._emit(
            "search_match_changed",
            result,
        )

        return result

    def previous_search_match(
        self,
    ) -> Optional[tuple[str, int]]:

        with self._lock:

            if not self._search_matches:
                return None

            self._current_search_match = (
                (
                    self._current_search_match
                    - 1
                )
                % len(
                    self._search_matches
                )
            )

            result = (
                self._search_matches[
                    self._current_search_match
                ]
            )

        self._emit(
            "search_match_changed",
            result,
        )

        return result

    # ========================================================================
    # VIEW
    # ========================================================================

    @property
    def view_mode(
        self,
    ) -> TerminalViewMode:

        with self._lock:
            return self._view_mode

    def set_view_mode(
        self,
        mode: TerminalViewMode,
    ) -> None:

        with self._lock:

            self._view_mode = mode

            self._fullscreen = (
                mode
                == TerminalViewMode.FULLSCREEN
            )

        self._emit(
            "view_mode_changed",
            mode,
        )

    def toggle_fullscreen(
        self,
    ) -> bool:

        with self._lock:

            self._fullscreen = (
                not self._fullscreen
            )

            self._view_mode = (
                TerminalViewMode.FULLSCREEN
                if self._fullscreen
                else TerminalViewMode.NORMAL
            )

            value = self._fullscreen

        self._emit(
            "fullscreen_changed",
            value,
        )

        return value

    def toggle_split(
        self,
    ) -> TerminalViewMode:

        with self._lock:

            if (
                self._view_mode
                == TerminalViewMode.SPLIT
            ):

                self._view_mode = (
                    TerminalViewMode.NORMAL
                )

            else:

                self._view_mode = (
                    TerminalViewMode.SPLIT
                )

            mode = self._view_mode

        self._emit(
            "view_mode_changed",
            mode,
        )

        return mode

    # ========================================================================
    # ZOOM / APPEARANCE
    # ========================================================================

    @property
    def font_size(self) -> float:

        with self._lock:
            return self._font_size

    def set_font_size(
        self,
        size: float,
    ) -> float:

        with self._lock:

            self._font_size = max(
                8.0,
                min(
                    48.0,
                    float(size),
                ),
            )

            value = self._font_size

        self._emit(
            "font_size_changed",
            value,
        )

        return value

    def zoom_in(
        self,
        amount: float = 1.0,
    ) -> float:

        return self.set_font_size(
            self.font_size + amount
        )

    def zoom_out(
        self,
        amount: float = 1.0,
    ) -> float:

        return self.set_font_size(
            self.font_size - amount
        )

    def reset_zoom(self) -> float:

        return self.set_font_size(
            14.0
        )

    @property
    def opacity(self) -> float:

        with self._lock:
            return self._opacity

    def set_opacity(
        self,
        value: float,
    ) -> float:

        with self._lock:

            self._opacity = max(
                0.25,
                min(
                    1.0,
                    float(value),
                ),
            )

            result = self._opacity

        self._emit(
            "opacity_changed",
            result,
        )

        return result

    # ========================================================================
    # CURSOR
    # ========================================================================

    @property
    def cursor_visible(
        self,
    ) -> bool:

        with self._lock:
            return self._cursor_visible

    def set_cursor_visible(
        self,
        visible: bool,
    ) -> None:

        with self._lock:

            self._cursor_visible = bool(
                visible
            )

        self._emit(
            "cursor_changed",
            self._cursor_visible,
        )

    # ========================================================================
    # ACTION DISPATCH
    # ========================================================================

    def execute_action(
        self,
        action: TerminalAction,
        **kwargs: Any,
    ) -> TerminalActionResult:

        if action == TerminalAction.NEW:

            session = self.create_session(
                shell=kwargs.get(
                    "shell"
                ),
                cwd=kwargs.get(
                    "cwd",
                    "",
                ),
            )

            return TerminalActionResult(
                action=action,
                success=True,
                message="Terminal session created.",
                data={
                    "session_id": session.id
                },
            )

        if action == TerminalAction.CLOSE:

            success = self.close_session(
                kwargs.get(
                    "session_id"
                )
            )

            return TerminalActionResult(
                action=action,
                success=success,
                message=(
                    "Terminal closed."
                    if success
                    else "Unable to close terminal."
                ),
            )

        if action == TerminalAction.CLEAR:

            self.clear()

            return TerminalActionResult(
                action=action,
                success=True,
                message="Terminal cleared.",
            )

        if action == TerminalAction.RUN:

            command = self.submit_command(
                kwargs.get(
                    "command"
                )
            )

            return TerminalActionResult(
                action=action,
                success=command is not None,
                message=(
                    "Command submitted."
                    if command
                    else "No command entered."
                ),
                data={
                    "command_id": (
                        command.id
                        if command
                        else None
                    )
                },
            )

        if action == TerminalAction.STOP:

            command_id = kwargs.get(
                "command_id"
            )

            if command_id:

                success = (
                    self.command_stopped(
                        command_id
                    )
                )

            else:

                session = (
                    self.get_active_session()
                )

                success = False

                if session:

                    with self._lock:

                        for command in reversed(
                            session.commands
                        ):

                            if (
                                command.state
                                == TerminalState.RUNNING
                            ):

                                command.state = (
                                    TerminalState.STOPPED
                                )

                                success = True

                                break

                        if success:

                            session.state = (
                                TerminalState.STOPPED
                            )

            self._emit(
                "stop_requested"
            )

            return TerminalActionResult(
                action=action,
                success=success,
                message=(
                    "Stop requested."
                    if success
                    else "No running command."
                ),
            )

        if action == TerminalAction.NEXT_TAB:

            session = self.next_session()

            return TerminalActionResult(
                action=action,
                success=session is not None,
                message="Next terminal selected.",
            )

        if action == TerminalAction.PREVIOUS_TAB:

            session = (
                self.previous_session()
            )

            return TerminalActionResult(
                action=action,
                success=session is not None,
                message="Previous terminal selected.",
            )

        if action == TerminalAction.SPLIT:

            mode = self.toggle_split()

            return TerminalActionResult(
                action=action,
                success=True,
                message=(
                    f"Terminal mode: "
                    f"{mode.value}"
                ),
            )

        if action == TerminalAction.TOGGLE_FULLSCREEN:

            value = self.toggle_fullscreen()

            return TerminalActionResult(
                action=action,
                success=True,
                message=(
                    "Fullscreen enabled."
                    if value
                    else "Fullscreen disabled."
                ),
            )

        if action == TerminalAction.ZOOM_IN:

            value = self.zoom_in(
                kwargs.get(
                    "amount",
                    1.0,
                )
            )

            return TerminalActionResult(
                action=action,
                success=True,
                message=(
                    f"Font size: "
                    f"{value:.1f}px"
                ),
                data={
                    "font_size": value
                },
            )

        if action == TerminalAction.ZOOM_OUT:

            value = self.zoom_out(
                kwargs.get(
                    "amount",
                    1.0,
                )
            )

            return TerminalActionResult(
                action=action,
                success=True,
                message=(
                    f"Font size: "
                    f"{value:.1f}px"
                ),
                data={
                    "font_size": value
                },
            )

        if action == TerminalAction.RESET_ZOOM:

            value = self.reset_zoom()

            return TerminalActionResult(
                action=action,
                success=True,
                message="Terminal zoom reset.",
                data={
                    "font_size": value
                },
            )

        if action == TerminalAction.SEARCH:

            count = self.search(
                kwargs.get(
                    "text",
                    "",
                )
            )

            return TerminalActionResult(
                action=action,
                success=True,
                message=(
                    f"{count} matches found."
                ),
                data={
                    "matches": count
                },
            )

        return TerminalActionResult(
            action=action,
            success=False,
            message="Unsupported terminal action.",
        )

    # ========================================================================
    # SNAPSHOT
    # ========================================================================

    def get_snapshot(
        self,
    ) -> dict[str, Any]:

        active = (
            self.get_active_session()
        )

        with self._lock:

            return {
                "running": self._running,
                "active_session_id": (
                    self._active_session_id
                ),
                "view_mode": (
                    self._view_mode.value
                ),
                "fullscreen": (
                    self._fullscreen
                ),
                "font_size": (
                    self._font_size
                ),
                "opacity": (
                    self._opacity
                ),
                "cursor_visible": (
                    self._cursor_visible
                ),
                "search": {
                    "text": self._search_text,
                    "matches": len(
                        self._search_matches
                    ),
                    "current": (
                        self._current_search_match
                    ),
                },
                "sessions": [
                    self._serialize_session(
                        session
                    )
                    for session
                    in self.get_sessions()
                ],
                "active_session": (
                    self._serialize_session(
                        active
                    )
                    if active
                    else None
                ),
            }

    @staticmethod
    def _serialize_session(
        session: Optional[TerminalSession],
    ) -> Optional[dict[str, Any]]:

        if session is None:
            return None

        return {
            "id": session.id,
            "title": session.title,
            "shell": session.shell,
            "cwd": session.cwd,
            "state": session.state.value,
            "input_text": session.input_text,
            "process_id": session.process_id,
            "created_at": session.created_at,
            "last_updated": (
                session.last_updated
            ),
            "line_count": len(
                session.lines
            ),
            "command_count": len(
                session.commands
            ),
            "history_count": len(
                session.history
            ),
        }

    # ========================================================================
    # STATUS
    # ========================================================================

    def status(self) -> dict[str, Any]:

        active = (
            self.get_active_session()
        )

        return {
            "running": self.running,
            "sessions": len(
                self._session_order
            ),
            "active_session_id": (
                self._active_session_id
            ),
            "active_state": (
                active.state.value
                if active
                else None
            ),
            "shell": (
                active.shell
                if active
                else None
            ),
            "cwd": (
                active.cwd
                if active
                else None
            ),
            "lines": (
                len(active.lines)
                if active
                else 0
            ),
            "commands": (
                len(active.commands)
                if active
                else 0
            ),
            "view_mode": (
                self._view_mode.value
            ),
            "fullscreen": (
                self._fullscreen
            ),
            "font_size": (
                self._font_size
            ),
        }

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
                "Event name cannot be empty."
            )

        if not callable(callback):
            raise TypeError(
                "Callback must be callable."
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

                # UI callbacks should never
                # crash RENIX.
                pass


# ============================================================================
# FACTORY
# ============================================================================


def create_terminal_ui() -> TerminalUI:

    ui = TerminalUI(
        default_shell="powershell",
        max_sessions=20,
        max_history=500,
        max_lines=10000,
    )

    ui.start()

    return ui


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "TerminalState",
    "TerminalAction",
    "TerminalViewMode",
    "TerminalOutputType",
    "TerminalLine",
    "TerminalCommand",
    "TerminalSession",
    "TerminalActionResult",
    "TerminalUI",
    "create_terminal_ui",
]


