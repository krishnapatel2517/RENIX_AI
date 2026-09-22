"""
RENIX Holographic UI - Animation Engine
=======================================

Central animation system for RENIX holographic UI.

Responsibilities:
    - Manage animation clips
    - Animate holographic UI elements
    - Position / rotation / scale animation
    - Opacity and glow animation
    - Easing functions
    - Looping
    - Ping-pong animations
    - Delays
    - Callbacks
    - Animation groups
    - Fade / pulse / slide / rotate effects

Renderer-independent.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


logger = logging.getLogger(
    "RENIX.holographic_ui.animation_engine"
)


# ============================================================================
# ENUMS
# ============================================================================


class AnimationState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Easing(str, Enum):
    LINEAR = "linear"

    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"

    EASE_IN_QUAD = "ease_in_quad"
    EASE_OUT_QUAD = "ease_out_quad"
    EASE_IN_OUT_QUAD = "ease_in_out_quad"

    EASE_IN_CUBIC = "ease_in_cubic"
    EASE_OUT_CUBIC = "ease_out_cubic"
    EASE_IN_OUT_CUBIC = "ease_in_out_cubic"

    EASE_IN_QUART = "ease_in_quart"
    EASE_OUT_QUART = "ease_out_quart"
    EASE_IN_OUT_QUART = "ease_in_out_quart"

    EASE_IN_BACK = "ease_in_back"
    EASE_OUT_BACK = "ease_out_back"
    EASE_IN_OUT_BACK = "ease_in_out_back"

    EASE_IN_ELASTIC = "ease_in_elastic"
    EASE_OUT_ELASTIC = "ease_out_elastic"

    EASE_IN_BOUNCE = "ease_in_bounce"
    EASE_OUT_BOUNCE = "ease_out_bounce"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class AnimationValue:
    """Single numeric animation value."""

    start: float
    end: float

    easing: Easing = Easing.EASE_IN_OUT


@dataclass
class AnimationClip:
    """
    Complete animation description.

    Target can be any object understood by the
    animation engine's property setter.
    """

    id: str

    target_id: str

    property_name: str

    start_value: float

    end_value: float

    duration: float = 1.0

    delay: float = 0.0

    easing: Easing = Easing.EASE_IN_OUT

    loops: int = 0

    ping_pong: bool = False

    state: AnimationState = AnimationState.IDLE

    elapsed: float = 0.0

    current_value: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    on_start: Optional[
        Callable[..., None]
    ] = None

    on_update: Optional[
        Callable[..., None]
    ] = None

    on_complete: Optional[
        Callable[..., None]
    ] = None

    on_cancel: Optional[
        Callable[..., None]
    ] = None


@dataclass
class AnimationGroup:
    """Collection of animations that can be controlled together."""

    id: str

    animation_ids: list[str] = field(
        default_factory=list
    )

    state: AnimationState = AnimationState.IDLE

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# ANIMATION ENGINE
# ============================================================================


class AnimationEngine:
    """
    Central animation controller.

    It does not draw anything itself.
    It updates target properties and emits events.
    """

    def __init__(
        self,
        *,
        hologram_engine: Any = None,
        hud_engine: Any = None,
    ) -> None:

        self.hologram_engine = (
            hologram_engine
        )

        self.hud_engine = hud_engine

        self._animations: dict[
            str,
            AnimationClip,
        ] = {}

        self._groups: dict[
            str,
            AnimationGroup,
        ] = {}

        self._callbacks: dict[
            str,
            list[Callable[..., None]],
        ] = {}

        self._lock = threading.RLock()

        self._running = False

        self._last_update = time.monotonic()

        self._time = 0.0

        logger.debug(
            "Animation engine created"
        )

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

        logger.info(
            "Animation engine started"
        )

    def stop(
        self,
        *,
        cancel_animations: bool = False,
    ) -> None:

        with self._lock:

            if not self._running:
                return

            self._running = False

        if cancel_animations:

            self.cancel_all()

        self._emit(
            "stopped",
            self,
        )

        logger.info(
            "Animation engine stopped"
        )

    # ========================================================================
    # UPDATE LOOP
    # ========================================================================

    def update(
        self,
        delta_time: Optional[float] = None,
    ) -> None:

        if not self._running:
            return

        now = time.monotonic()

        if delta_time is None:

            delta_time = (
                now - self._last_update
            )

        delta_time = max(
            0.0,
            min(
                0.25,
                float(delta_time),
            ),
        )

        self._last_update = now

        with self._lock:

            self._time += delta_time

            animations = list(
                self._animations.values()
            )

        for animation in animations:

            if (
                animation.state
                == AnimationState.RUNNING
            ):

                self._update_animation(
                    animation,
                    delta_time,
                )

        self._update_groups()

        self._emit(
            "updated",
            delta_time,
        )

    # ========================================================================
    # CREATE ANIMATION
    # ========================================================================

    def create(
        self,
        target_id: str,
        property_name: str,
        start_value: float,
        end_value: float,
        *,
        duration: float = 1.0,
        delay: float = 0.0,
        easing: Easing | str = Easing.EASE_IN_OUT,
        loops: int = 0,
        ping_pong: bool = False,
        animation_id: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
        on_start: Optional[
            Callable[..., None]
        ] = None,
        on_update: Optional[
            Callable[..., None]
        ] = None,
        on_complete: Optional[
            Callable[..., None]
        ] = None,
        on_cancel: Optional[
            Callable[..., None]
        ] = None,
    ) -> AnimationClip:

        if not target_id:

            raise ValueError(
                "target_id cannot be empty"
            )

        if not property_name:

            raise ValueError(
                "property_name cannot be empty"
            )

        duration = max(
            0.0001,
            float(duration),
        )

        delay = max(
            0.0,
            float(delay),
        )

        if isinstance(
            easing,
            str,
        ):

            try:

                easing = Easing(
                    easing.lower()
                )

            except ValueError:

                easing = Easing.EASE_IN_OUT

        animation = AnimationClip(
            id=(
                animation_id
                or uuid.uuid4().hex
            ),
            target_id=target_id,
            property_name=property_name,
            start_value=float(
                start_value
            ),
            end_value=float(
                end_value
            ),
            duration=duration,
            delay=delay,
            easing=easing,
            loops=max(
                0,
                int(loops),
            ),
            ping_pong=bool(
                ping_pong
            ),
            current_value=float(
                start_value
            ),
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
            on_start=on_start,
            on_update=on_update,
            on_complete=on_complete,
            on_cancel=on_cancel,
        )

        with self._lock:

            self._animations[
                animation.id
            ] = animation

        self._emit(
            "created",
            animation,
        )

        return animation

    # ========================================================================
    # PLAY
    # ========================================================================

    def play(
        self,
        animation_id: str,
    ) -> bool:

        animation = self.get(
            animation_id
        )

        if animation is None:
            return False

        with self._lock:

            if (
                animation.state
                == AnimationState.RUNNING
            ):

                return True

            if (
                animation.state
                == AnimationState.COMPLETED
            ):

                animation.elapsed = 0.0

                animation.current_value = (
                    animation.start_value
                )

            animation.state = (
                AnimationState.RUNNING
            )

        if animation.on_start:

            self._safe_callback(
                animation.on_start,
                animation,
            )

        self._emit(
            "started",
            animation,
        )

        return True

    # ========================================================================
    # PAUSE
    # ========================================================================

    def pause(
        self,
        animation_id: str,
    ) -> bool:

        animation = self.get(
            animation_id
        )

        if animation is None:
            return False

        with self._lock:

            if (
                animation.state
                != AnimationState.RUNNING
            ):

                return False

            animation.state = (
                AnimationState.PAUSED
            )

        self._emit(
            "paused",
            animation,
        )

        return True

    # ========================================================================
    # RESUME
    # ========================================================================

    def resume(
        self,
        animation_id: str,
    ) -> bool:

        animation = self.get(
            animation_id
        )

        if animation is None:
            return False

        with self._lock:

            if (
                animation.state
                != AnimationState.PAUSED
            ):

                return False

            animation.state = (
                AnimationState.RUNNING
            )

        self._emit(
            "resumed",
            animation,
        )

        return True

    # ========================================================================
    # CANCEL
    # ========================================================================

    def cancel(
        self,
        animation_id: str,
    ) -> bool:

        animation = self.get(
            animation_id
        )

        if animation is None:
            return False

        with self._lock:

            animation.state = (
                AnimationState.CANCELLED
            )

        if animation.on_cancel:

            self._safe_callback(
                animation.on_cancel,
                animation,
            )

        self._emit(
            "cancelled",
            animation,
        )

        return True

    def cancel_all(self) -> None:

        with self._lock:

            animations = list(
                self._animations.values()
            )

        for animation in animations:

            if animation.state in (
                AnimationState.RUNNING,
                AnimationState.PAUSED,
            ):

                self.cancel(
                    animation.id
                )

    # ========================================================================
    # REMOVE
    # ========================================================================

    def remove(
        self,
        animation_id: str,
    ) -> bool:

        with self._lock:

            animation = self._animations.pop(
                animation_id,
                None,
            )

        if animation is None:
            return False

        # Remove from groups.
        with self._lock:

            for group in self._groups.values():

                if animation_id in group.animation_ids:

                    group.animation_ids.remove(
                        animation_id
                    )

        self._emit(
            "removed",
            animation,
        )

        return True

    # ========================================================================
    # GETTERS
    # ========================================================================

    def get(
        self,
        animation_id: str,
    ) -> Optional[AnimationClip]:

        with self._lock:

            return self._animations.get(
                animation_id
            )

    def get_all(self) -> list[AnimationClip]:

        with self._lock:

            return list(
                self._animations.values()
            )

    def get_running(
        self,
    ) -> list[AnimationClip]:

        with self._lock:

            return [
                animation
                for animation
                in self._animations.values()
                if animation.state
                == AnimationState.RUNNING
            ]

    # ========================================================================
    # INTERNAL ANIMATION UPDATE
    # ========================================================================

    def _update_animation(
        self,
        animation: AnimationClip,
        delta_time: float,
    ) -> None:

        animation.elapsed += delta_time

        # Delay phase.
        if (
            animation.elapsed
            < animation.delay
        ):

            return

        active_time = (
            animation.elapsed
            - animation.delay
        )

        duration = animation.duration

        cycle = int(
            active_time // duration
        )

        cycle_progress = (
            active_time % duration
        ) / duration

        total_cycles = (
            animation.loops + 1
        )

        if cycle >= total_cycles:

            cycle = total_cycles - 1

            cycle_progress = 1.0

            final_value = (
                animation.end_value
                if not animation.ping_pong
                or animation.loops % 2 == 0
                else animation.start_value
            )

            animation.current_value = (
                final_value
            )

            self._apply_value(
                animation,
                final_value,
            )

            with self._lock:

                animation.state = (
                    AnimationState.COMPLETED
                )

            if animation.on_complete:

                self._safe_callback(
                    animation.on_complete,
                    animation,
                )

            self._emit(
                "completed",
                animation,
            )

            return

        if (
            animation.ping_pong
            and cycle % 2 == 1
        ):

            cycle_progress = (
                1.0 - cycle_progress
            )

        eased = self.ease(
            cycle_progress,
            animation.easing,
        )

        value = (
            animation.start_value
            + (
                animation.end_value
                - animation.start_value
            )
            * eased
        )

        animation.current_value = value

        self._apply_value(
            animation,
            value,
        )

        if animation.on_update:

            self._safe_callback(
                animation.on_update,
                animation,
                value,
            )

        self._emit(
            "updated_animation",
            animation,
            value,
        )

    # ========================================================================
    # APPLY VALUE
    # ========================================================================

    def _apply_value(
        self,
        animation: AnimationClip,
        value: float,
    ) -> None:

        target_id = animation.target_id

        property_name = (
            animation.property_name
        )

        # Hologram engine.
        if self.hologram_engine is not None:

            try:

                if property_name == "opacity":

                    self.hologram_engine.set_opacity(
                        target_id,
                        value,
                    )

                    return

                if property_name == "glow":

                    self.hologram_engine.set_glow(
                        target_id,
                        value,
                    )

                    return

                if property_name == "scale":

                    self.hologram_engine.set_scale(
                        target_id,
                        value,
                    )

                    return

                if property_name == "rotation_z":

                    self.hologram_engine.set_rotation(
                        target_id,
                        z=value,
                    )

                    return

                if property_name == "rotation_x":

                    self.hologram_engine.set_rotation(
                        target_id,
                        x=value,
                    )

                    return

                if property_name == "rotation_y":

                    self.hologram_engine.set_rotation(
                        target_id,
                        y=value,
                    )

                    return

                if property_name == "x":

                    hologram = (
                        self.hologram_engine.get(
                            target_id
                        )
                    )

                    if hologram is not None:

                        self.hologram_engine.set_position(
                            target_id,
                            value,
                            hologram.transform.y,
                            hologram.transform.z,
                        )

                    return

                if property_name == "y":

                    hologram = (
                        self.hologram_engine.get(
                            target_id
                        )
                    )

                    if hologram is not None:

                        self.hologram_engine.set_position(
                            target_id,
                            hologram.transform.x,
                            value,
                            hologram.transform.z,
                        )

                    return

                if property_name == "z":

                    hologram = (
                        self.hologram_engine.get(
                            target_id
                        )
                    )

                    if hologram is not None:

                        self.hologram_engine.set_position(
                            target_id,
                            hologram.transform.x,
                            hologram.transform.y,
                            value,
                        )

                    return

            except Exception:

                logger.exception(
                    "Failed applying hologram animation"
                )

        # HUD engine.
        if self.hud_engine is not None:

            try:

                if property_name == "opacity":

                    self.hud_engine.set_opacity(
                        target_id,
                        value,
                    )

                    return

                if property_name == "glow":

                    self.hud_engine.set_glow(
                        target_id,
                        value,
                    )

                    return

                if property_name == "scale":

                    self.hud_engine.set_scale(
                        target_id,
                        value,
                    )

                    return

                if property_name == "x":

                    element = (
                        self.hud_engine.get(
                            target_id
                        )
                    )

                    if element is not None:

                        self.hud_engine.set_position(
                            target_id,
                            value,
                            element.position.y,
                        )

                    return

                if property_name == "y":

                    element = (
                        self.hud_engine.get(
                            target_id
                        )
                    )

                    if element is not None:

                        self.hud_engine.set_position(
                            target_id,
                            element.position.x,
                            value,
                        )

                    return

            except Exception:

                logger.exception(
                    "Failed applying HUD animation"
                )

    # ========================================================================
    # GROUPS
    # ========================================================================

    def create_group(
        self,
        animation_ids: Optional[
            list[str]
        ] = None,
        *,
        group_id: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> AnimationGroup:

        group = AnimationGroup(
            id=(
                group_id
                or uuid.uuid4().hex
            ),
            animation_ids=list(
                animation_ids or []
            ),
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
        )

        with self._lock:

            self._groups[
                group.id
            ] = group

        self._emit(
            "group_created",
            group,
        )

        return group

    def add_to_group(
        self,
        group_id: str,
        animation_id: str,
    ) -> bool:

        with self._lock:

            group = self._groups.get(
                group_id
            )

            animation = self._animations.get(
                animation_id
            )

            if group is None or animation is None:

                return False

            if (
                animation_id
                not in group.animation_ids
            ):

                group.animation_ids.append(
                    animation_id
                )

        return True

    def play_group(
        self,
        group_id: str,
    ) -> bool:

        with self._lock:

            group = self._groups.get(
                group_id
            )

        if group is None:
            return False

        success = False

        for animation_id in (
            group.animation_ids
        ):

            if self.play(animation_id):

                success = True

        group.state = (
            AnimationState.RUNNING
        )

        self._emit(
            "group_started",
            group,
        )

        return success

    def pause_group(
        self,
        group_id: str,
    ) -> bool:

        with self._lock:

            group = self._groups.get(
                group_id
            )

        if group is None:
            return False

        success = False

        for animation_id in (
            group.animation_ids
        ):

            if self.pause(animation_id):

                success = True

        group.state = (
            AnimationState.PAUSED
        )

        self._emit(
            "group_paused",
            group,
        )

        return success

    def cancel_group(
        self,
        group_id: str,
    ) -> bool:

        with self._lock:

            group = self._groups.get(
                group_id
            )

        if group is None:
            return False

        for animation_id in (
            group.animation_ids
        ):

            self.cancel(animation_id)

        group.state = (
            AnimationState.CANCELLED
        )

        self._emit(
            "group_cancelled",
            group,
        )

        return True

    def _update_groups(self) -> None:

        with self._lock:

            groups = list(
                self._groups.values()
            )

        for group in groups:

            if not group.animation_ids:
                continue

            animations = [
                self.get(animation_id)
                for animation_id
                in group.animation_ids
            ]

            animations = [
                animation
                for animation
                in animations
                if animation is not None
            ]

            if not animations:
                continue

            if all(
                animation.state
                == AnimationState.COMPLETED
                for animation in animations
            ):

                if (
                    group.state
                    != AnimationState.COMPLETED
                ):

                    group.state = (
                        AnimationState.COMPLETED
                    )

                    self._emit(
                        "group_completed",
                        group,
                    )

    # ========================================================================
    # EASING
    # ========================================================================

    @staticmethod
    def ease(
        t: float,
        easing: Easing | str,
    ) -> float:

        t = max(
            0.0,
            min(
                1.0,
                float(t),
            ),
        )

        if isinstance(
            easing,
            str,
        ):

            try:

                easing = Easing(
                    easing.lower()
                )

            except ValueError:

                easing = Easing.LINEAR

        if easing == Easing.LINEAR:
            return t

        if easing == Easing.EASE_IN:
            return t * t

        if easing == Easing.EASE_OUT:
            return 1.0 - (1.0 - t) ** 2

        if easing == Easing.EASE_IN_OUT:

            if t < 0.5:

                return 2.0 * t * t

            return 1.0 - (
                (-2.0 * t + 2.0) ** 2
            ) / 2.0

        if easing == Easing.EASE_IN_QUAD:
            return t * t

        if easing == Easing.EASE_OUT_QUAD:
            return 1.0 - (1.0 - t) ** 2

        if easing == Easing.EASE_IN_OUT_QUAD:

            if t < 0.5:
                return 2 * t * t

            return 1 - (
                (-2 * t + 2) ** 2
            ) / 2

        if easing == Easing.EASE_IN_CUBIC:
            return t ** 3

        if easing == Easing.EASE_OUT_CUBIC:
            return 1 - (1 - t) ** 3

        if easing == Easing.EASE_IN_OUT_CUBIC:

            if t < 0.5:
                return 4 * t ** 3

            return 1 - (
                (-2 * t + 2) ** 3
            ) / 2

        if easing == Easing.EASE_IN_QUART:
            return t ** 4

        if easing == Easing.EASE_OUT_QUART:
            return 1 - (1 - t) ** 4

        if easing == Easing.EASE_IN_OUT_QUART:

            if t < 0.5:
                return 8 * t ** 4

            return 1 - (
                (-2 * t + 2) ** 4
            ) / 2

        if easing == Easing.EASE_IN_BACK:

            c1 = 1.70158
            c3 = c1 + 1

            return (
                c3 * t ** 3
                - c1 * t ** 2
            )

        if easing == Easing.EASE_OUT_BACK:

            c1 = 1.70158
            c3 = c1 + 1

            return (
                1
                + c3 * (t - 1) ** 3
                + c1 * (t - 1) ** 2
            )

        if easing == Easing.EASE_IN_OUT_BACK:

            c1 = 1.70158
            c2 = c1 * 1.525

            if t < 0.5:

                return (
                    (
                        (2 * t) ** 2
                        * (
                            (c2 + 1)
                            * 2
                            * t
                            - c2
                        )
                    )
                    / 2
                )

            return (
                (
                    (2 * t - 2) ** 2
                    * (
                        (c2 + 1)
                        * (t * 2 - 2)
                        + c2
                    )
                    + 2
                )
                / 2
            )

        if easing == Easing.EASE_IN_ELASTIC:

            if t == 0 or t == 1:
                return t

            c4 = (
                2 * math.pi / 3
            )

            return -(
                2 ** (10 * t - 10)
                * math.sin(
                    (t * 10 - 10.75)
                    * c4
                )
            )

        if easing == Easing.EASE_OUT_ELASTIC:

            if t == 0 or t == 1:
                return t

            c4 = (
                2 * math.pi / 3
            )

            return (
                2 ** (-10 * t)
                * math.sin(
                    (t * 10 - 0.75)
                    * c4
                )
                + 1
            )

        if easing == Easing.EASE_OUT_BOUNCE:

            return AnimationEngine._bounce_out(
                t
            )

        if easing == Easing.EASE_IN_BOUNCE:

            return 1 - AnimationEngine._bounce_out(
                1 - t
            )

        return t

    @staticmethod
    def _bounce_out(
        t: float,
    ) -> float:

        n1 = 7.5625
        d1 = 2.75

        if t < 1 / d1:

            return n1 * t * t

        if t < 2 / d1:

            t -= 1.5 / d1

            return n1 * t * t + 0.75

        if t < 2.5 / d1:

            t -= 2.25 / d1

            return n1 * t * t + 0.9375

        t -= 2.625 / d1

        return n1 * t * t + 0.984375

    # ========================================================================
    # PRESET ANIMATIONS
    # ========================================================================

    def fade_in(
        self,
        target_id: str,
        *,
        duration: float = 0.35,
        start: float = 0.0,
        end: float = 1.0,
        property_name: str = "opacity",
        auto_play: bool = True,
    ) -> AnimationClip:

        animation = self.create(
            target_id,
            property_name,
            start,
            end,
            duration=duration,
            easing=Easing.EASE_OUT,
        )

        if auto_play:
            self.play(animation.id)

        return animation

    def fade_out(
        self,
        target_id: str,
        *,
        duration: float = 0.35,
        start: float = 1.0,
        end: float = 0.0,
        property_name: str = "opacity",
        auto_play: bool = True,
    ) -> AnimationClip:

        animation = self.create(
            target_id,
            property_name,
            start,
            end,
            duration=duration,
            easing=Easing.EASE_IN,
        )

        if auto_play:
            self.play(animation.id)

        return animation

    def pulse(
        self,
        target_id: str,
        *,
        property_name: str = "glow",
        minimum: float = 0.7,
        maximum: float = 1.5,
        duration: float = 0.7,
        loops: int = 3,
        auto_play: bool = True,
    ) -> AnimationClip:

        animation = self.create(
            target_id,
            property_name,
            minimum,
            maximum,
            duration=duration,
            easing=Easing.EASE_IN_OUT,
            loops=loops,
            ping_pong=True,
        )

        if auto_play:
            self.play(animation.id)

        return animation

    def rotate(
        self,
        target_id: str,
        *,
        degrees: float = 360.0,
        duration: float = 2.0,
        loops: int = 0,
        property_name: str = "rotation_z",
        auto_play: bool = True,
    ) -> AnimationClip:

        target = 0.0

        if self.hologram_engine is not None:

            hologram = (
                self.hologram_engine.get(
                    target_id
                )
            )

            if hologram is not None:

                if property_name == "rotation_x":
                    target = (
                        hologram.transform.rotation_x
                    )

                elif property_name == "rotation_y":
                    target = (
                        hologram.transform.rotation_y
                    )

                else:
                    target = (
                        hologram.transform.rotation_z
                    )

        animation = self.create(
            target_id,
            property_name,
            target,
            target + degrees,
            duration=duration,
            easing=Easing.LINEAR,
            loops=loops,
        )

        if auto_play:
            self.play(animation.id)

        return animation

    # ========================================================================
    # CALLBACKS
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

            self._safe_callback(
                callback,
                *args,
                **kwargs,
            )

    @staticmethod
    def _safe_callback(
        callback: Callable[..., None],
        *args: Any,
        **kwargs: Any,
    ) -> None:

        try:

            callback(
                *args,
                **kwargs,
            )

        except Exception:

            logger.exception(
                "Animation callback failed"
            )

    # ========================================================================
    # STATUS
    # ========================================================================

    @property
    def running(self) -> bool:

        with self._lock:

            return self._running

    def to_dict(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "running": self._running,
                "animations": len(
                    self._animations
                ),
                "running_animations": len(
                    [
                        animation
                        for animation
                        in self._animations.values()
                        if animation.state
                        == AnimationState.RUNNING
                    ]
                ),
                "groups": len(
                    self._groups
                ),
            }


__all__ = [
    "AnimationState",
    "Easing",
    "AnimationValue",
    "AnimationClip",
    "AnimationGroup",
    "AnimationEngine",
]


