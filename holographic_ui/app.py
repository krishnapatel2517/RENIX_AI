"""
RENIX Holographic UI - Application Controller
==============================================

Top-level application lifecycle for the RENIX holographic interface.

Responsibilities:
    - Initialize the holographic UI subsystems
    - Coordinate the UI lifecycle
    - Connect gesture and voice interfaces
    - Manage the render/update loop
    - Handle shutdown
    - Publish UI state/events

This file does NOT implement rendering itself.
Rendering belongs to hologram_engine.py.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


logger = logging.getLogger("RENIX.holographic_ui.app")


# ============================================================================
# ENUMS
# ============================================================================


class AppState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class AppConfig:
    """Runtime configuration for the holographic application."""

    target_fps: int = 60

    update_interval: float = 1.0 / 60.0

    auto_start: bool = False

    enable_gestures: bool = True

    enable_voice: bool = True

    enable_hologram: bool = True

    enable_hud: bool = True

    enable_particles: bool = True

    enable_3d: bool = True

    enable_notifications: bool = True

    enable_widgets: bool = True

    background_update: bool = True

    graceful_shutdown_timeout: float = 5.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# APPLICATION
# ============================================================================


class RENIXApp:
    """
    Main controller for the RENIX holographic UI.

    The application coordinates all UI subsystems without placing their
    implementation details inside this class.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        *,
        core: Any = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else AppConfig()
        )

        self.core = core

        self.state = AppState.CREATED

        self.started_at: Optional[float] = None

        self.stopped_at: Optional[float] = None

        self.frame_count = 0

        self.last_frame_time = 0.0

        self.current_fps = 0.0

        self._running = False

        self._shutdown_requested = False

        self._thread: Optional[
            threading.Thread
        ] = None

        self._lock = threading.RLock()

        self._stop_event = threading.Event()

        self._components: dict[
            str,
            Any,
        ] = {}

        self._callbacks: dict[
            str,
            list[Callable[..., None]],
        ] = {}

        self._last_error: Optional[
            Exception
        ] = None

        logger.debug(
            "RENIX holographic application created"
        )

        if self.config.auto_start:

            self.start()

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self) -> bool:
        """
        Start the holographic application.

        Returns:
            bool: True when startup succeeds.
        """

        with self._lock:

            if self.state == AppState.RUNNING:

                return True

            if self.state == AppState.STARTING:

                return False

            self.state = AppState.STARTING

            self._shutdown_requested = False

            self._stop_event.clear()

            try:

                self._initialize_components()

                self._running = True

                self.started_at = time.time()

                self.stopped_at = None

                self.state = AppState.RUNNING

                self._emit(
                    "started",
                    self,
                )

                logger.info(
                    "RENIX holographic UI started"
                )

                if self.config.background_update:

                    self._thread = threading.Thread(
                        target=self._run_loop,
                        name="RENIX-HolographicUI",
                        daemon=True,
                    )

                    self._thread.start()

                return True

            except Exception as exc:

                self._last_error = exc

                self.state = AppState.ERROR

                logger.exception(
                    "Failed to start holographic UI"
                )

                self._emit(
                    "error",
                    exc,
                )

                return False

    def stop(self) -> bool:
        """
        Gracefully stop the application.
        """

        with self._lock:

            if self.state == AppState.STOPPED:

                return True

            if self.state == AppState.STOPPING:

                return False

            self.state = AppState.STOPPING

            self._running = False

            self._shutdown_requested = True

            self._stop_event.set()

        try:

            self._shutdown_components()

        except Exception as exc:

            self._last_error = exc

            logger.exception(
                "Error while shutting down UI"
            )

        thread = self._thread

        if (
            thread is not None
            and thread.is_alive()
            and thread
            is not threading.current_thread()
        ):

            thread.join(
                timeout=self.config
                .graceful_shutdown_timeout
            )

        with self._lock:

            self.stopped_at = time.time()

            self.state = AppState.STOPPED

            self._emit(
                "stopped",
                self,
            )

        logger.info(
            "RENIX holographic UI stopped"
        )

        return True

    def pause(self) -> bool:
        """
        Pause UI updates without destroying components.
        """

        with self._lock:

            if self.state != AppState.RUNNING:

                return False

            self.state = AppState.PAUSED

            self._emit(
                "paused",
                self,
            )

            return True

    def resume(self) -> bool:
        """
        Resume a paused UI.
        """

        with self._lock:

            if self.state != AppState.PAUSED:

                return False

            self.state = AppState.RUNNING

            self._emit(
                "resumed",
                self,
            )

            return True

    # ========================================================================
    # COMPONENT INITIALIZATION
    # ========================================================================

    def _initialize_components(self) -> None:
        """
        Initialize UI subsystems.

        Components are imported lazily so that importing RENIX does not
        immediately require every graphical dependency.
        """

        self._components.clear()

        if self.config.enable_hologram:

            try:

                from .hologram_engine import (
                    HologramEngine,
                )

                self._components[
                    "hologram_engine"
                ] = HologramEngine()

            except ImportError:

                logger.warning(
                    "Hologram engine unavailable"
                )

        if self.config.enable_hud:

            try:

                from .hud_engine import (
                    HUDEngine,
                )

                self._components[
                    "hud_engine"
                ] = HUDEngine()

            except ImportError:

                logger.warning(
                    "HUD engine unavailable"
                )

        if self.config.enable_particles:

            try:

                from .particle_engine import (
                    ParticleEngine,
                )

                self._components[
                    "particle_engine"
                ] = ParticleEngine()

            except ImportError:

                logger.warning(
                    "Particle engine unavailable"
                )

        if self.config.enable_3d:

            try:
                import importlib

                module = importlib.import_module(
                    f"{__package__}.3d_engine"
                )
                self._components["3d_engine"] = module.Engine3D()

            except Exception as exc:
                logger.warning("3D engine unavailable: %s", exc)

        if self.config.enable_gestures:

            try:

                from .gesture_interface import (
                    GestureInterface,
                )

                self._components[
                    "gesture_interface"
                ] = GestureInterface()

            except ImportError:

                logger.warning(
                    "Gesture interface unavailable"
                )

        if self.config.enable_voice:

            try:

                from .voice_interface import (
                    VoiceInterface,
                )

                self._components[
                    "voice_interface"
                ] = VoiceInterface()

            except ImportError:

                logger.warning(
                    "Voice interface unavailable"
                )

        if self.config.enable_notifications:

            try:

                from .notification_ui import (
                    NotificationUI,
                )

                self._components[
                    "notification_ui"
                ] = NotificationUI()

            except ImportError:

                logger.warning(
                    "Notification UI unavailable"
                )

        self._start_components()

    def _start_components(self) -> None:

        for name, component in list(
            self._components.items()
        ):

            start = getattr(
                component,
                "start",
                None,
            )

            if callable(start):

                try:

                    start()

                except Exception:

                    logger.exception(
                        "Failed to start UI component: %s",
                        name,
                    )

    def _shutdown_components(self) -> None:

        for name, component in reversed(
            list(self._components.items())
        ):

            stop = getattr(
                component,
                "stop",
                None,
            )

            if callable(stop):

                try:

                    stop()

                except Exception:

                    logger.exception(
                        "Failed to stop UI component: %s",
                        name,
                    )

        self._components.clear()

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    def _run_loop(self) -> None:
        """
        Background application update loop.
        """

        previous = time.perf_counter()

        while not self._stop_event.is_set():

            loop_start = time.perf_counter()

            with self._lock:

                state = self.state

            if state == AppState.RUNNING:

                now = time.perf_counter()

                delta_time = max(
                    0.0,
                    now - previous,
                )

                previous = now

                try:

                    self.update(
                        delta_time
                    )

                    self.render(
                        delta_time
                    )

                except Exception as exc:

                    self._last_error = exc

                    logger.exception(
                        "Error in holographic UI loop"
                    )

                    self._emit(
                        "error",
                        exc,
                    )

            elapsed = (
                time.perf_counter()
                - loop_start
            )

            sleep_time = max(
                0.0,
                self.config.update_interval
                - elapsed,
            )

            self._stop_event.wait(
                sleep_time
            )

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        delta_time: float,
    ) -> None:
        """
        Update all active UI components.
        """

        if self.state != AppState.RUNNING:

            return

        delta_time = max(
            0.0,
            float(delta_time),
        )

        frame_start = time.perf_counter()

        for name, component in list(
            self._components.items()
        ):

            update = getattr(
                component,
                "update",
                None,
            )

            if not callable(update):

                continue

            try:

                update(delta_time)

            except Exception:

                logger.exception(
                    "Update failed for component: %s",
                    name,
                )

        self.frame_count += 1

        now = time.perf_counter()

        self.last_frame_time = (
            now - frame_start
        )

        if self.last_frame_time > 0:

            instant_fps = (
                1.0
                / self.last_frame_time
            )

            if self.current_fps <= 0:

                self.current_fps = (
                    instant_fps
                )

            else:

                self.current_fps = (
                    self.current_fps * 0.9
                    + instant_fps * 0.1
                )

        self._emit(
            "updated",
            delta_time,
        )

    # ========================================================================
    # RENDER
    # ========================================================================

    def render(
        self,
        delta_time: float,
    ) -> None:
        """
        Render all active components.
        """

        if self.state != AppState.RUNNING:

            return

        for name, component in list(
            self._components.items()
        ):

            render = getattr(
                component,
                "render",
                None,
            )

            if not callable(render):

                continue

            try:

                render(delta_time)

            except Exception:

                logger.exception(
                    "Render failed for component: %s",
                    name,
                )

        self._emit(
            "rendered",
            delta_time,
        )

    # ========================================================================
    # COMPONENT ACCESS
    # ========================================================================

    def get_component(
        self,
        name: str,
        default: Any = None,
    ) -> Any:

        with self._lock:

            return self._components.get(
                name,
                default,
            )

    def register_component(
        self,
        name: str,
        component: Any,
    ) -> None:

        if not name:

            raise ValueError(
                "Component name cannot be empty"
            )

        if component is None:

            raise ValueError(
                "Component cannot be None"
            )

        with self._lock:

            self._components[name] = (
                component
            )

        start = getattr(
            component,
            "start",
            None,
        )

        if (
            callable(start)
            and self.state == AppState.RUNNING
        ):

            start()

    def remove_component(
        self,
        name: str,
    ) -> Any:

        with self._lock:

            component = self._components.pop(
                name,
                None,
            )

        if component is not None:

            stop = getattr(
                component,
                "stop",
                None,
            )

            if callable(stop):

                try:

                    stop()

                except Exception:

                    logger.exception(
                        "Failed to stop removed component: %s",
                        name,
                    )

        return component

    def components(self) -> dict[str, Any]:

        with self._lock:

            return dict(
                self._components
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
                "Event name cannot be empty"
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
                    "UI event callback failed: %s",
                    event,
                )

    # ========================================================================
    # STATUS
    # ========================================================================

    def is_running(self) -> bool:

        return self.state == AppState.RUNNING

    def is_paused(self) -> bool:

        return self.state == AppState.PAUSED

    def is_stopped(self) -> bool:

        return self.state == AppState.STOPPED

    def get_state(self) -> AppState:

        return self.state

    def get_uptime(self) -> float:

        if self.started_at is None:

            return 0.0

        end = (
            self.stopped_at
            if self.stopped_at is not None
            else time.time()
        )

        return max(
            0.0,
            end - self.started_at,
        )

    def get_status(self) -> dict[str, Any]:

        return {
            "state": self.state.value,
            "running": self.is_running(),
            "paused": self.is_paused(),
            "frame_count": self.frame_count,
            "fps": self.current_fps,
            "frame_time": self.last_frame_time,
            "uptime": self.get_uptime(),
            "components": list(
                self._components.keys()
            ),
            "component_count": len(
                self._components
            ),
            "error": (
                str(self._last_error)
                if self._last_error
                else None
            ),
        }

    # ========================================================================
    # CONTEXT MANAGER
    # ========================================================================

    def __enter__(self) -> "RENIXApp":

        self.start()

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:

        self.stop()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def create_app(
    config: Optional[AppConfig] = None,
    *,
    core: Any = None,
) -> RENIXApp:
    """
    Create a RENIX holographic application instance.
    """

    return RENIXApp(
        config=config,
        core=core,
    )


__all__ = [
    "AppState",
    "AppConfig",
    "RENIXApp",
    "create_app",
]


