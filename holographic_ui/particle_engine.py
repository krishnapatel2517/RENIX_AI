"""
RENIX Holographic UI
Particle Engine
==================

Handles holographic particles used by RENIX:

- Floating particles
- Ambient particles
- Burst effects
- Trails
- Sparks
- Energy fields
- Cursor/gesture particles
- Spawn/despawn animations
- Particle emitters
- Particle groups
- Performance limits
- Deterministic update loop

Renderer-independent design.

The engine generates and updates particle state.
The actual renderer can consume the particle snapshots.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
import uuid

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


logger = logging.getLogger(
    "RENIX.holographic_ui.particle_engine"
)


# ============================================================================
# ENUMS
# ============================================================================


class ParticleState(str, Enum):
    ACTIVE = "active"
    DEAD = "dead"
    PAUSED = "paused"


class EmitterShape(str, Enum):
    POINT = "point"
    CIRCLE = "circle"
    SPHERE = "sphere"
    BOX = "box"
    RING = "ring"
    LINE = "line"


class ParticleBlendMode(str, Enum):
    NORMAL = "normal"
    ADDITIVE = "additive"
    SCREEN = "screen"


# ============================================================================
# VECTOR
# ============================================================================


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def copy(self) -> "Vector3":
        return Vector3(
            self.x,
            self.y,
            self.z,
        )

    def __add__(
        self,
        other: "Vector3",
    ) -> "Vector3":

        return Vector3(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(
        self,
        other: "Vector3",
    ) -> "Vector3":

        return Vector3(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __mul__(
        self,
        scalar: float,
    ) -> "Vector3":

        return Vector3(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
        )

    def length(self) -> float:

        return math.sqrt(
            self.x * self.x
            + self.y * self.y
            + self.z * self.z
        )

    def normalized(self) -> "Vector3":

        length = self.length()

        if length <= 0.000001:

            return Vector3()

        return Vector3(
            self.x / length,
            self.y / length,
            self.z / length,
        )


# ============================================================================
# COLOR
# ============================================================================


@dataclass
class ParticleColor:
    r: float = 0.0
    g: float = 1.0
    b: float = 0.65
    a: float = 1.0

    def copy(self) -> "ParticleColor":

        return ParticleColor(
            self.r,
            self.g,
            self.b,
            self.a,
        )


# ============================================================================
# PARTICLE
# ============================================================================


@dataclass
class Particle:
    id: str

    position: Vector3

    velocity: Vector3

    acceleration: Vector3 = field(
        default_factory=Vector3
    )

    color: ParticleColor = field(
        default_factory=ParticleColor
    )

    start_color: ParticleColor = field(
        default_factory=ParticleColor
    )

    end_color: ParticleColor = field(
        default_factory=ParticleColor
    )

    size: float = 1.0

    start_size: float = 1.0

    end_size: float = 0.0

    opacity: float = 1.0

    start_opacity: float = 1.0

    end_opacity: float = 0.0

    lifetime: float = 1.0

    age: float = 0.0

    rotation: float = 0.0

    rotation_speed: float = 0.0

    state: ParticleState = (
        ParticleState.ACTIVE
    )

    gravity: float = 0.0

    drag: float = 0.0

    glow: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def progress(self) -> float:

        if self.lifetime <= 0:
            return 1.0

        return max(
            0.0,
            min(
                1.0,
                self.age / self.lifetime,
            ),
        )

    @property
    def alive(self) -> bool:

        return (
            self.state
            == ParticleState.ACTIVE
            and self.age < self.lifetime
        )


# ============================================================================
# EMITTER CONFIGURATION
# ============================================================================


@dataclass
class EmitterConfig:
    name: str = "default"

    shape: EmitterShape = (
        EmitterShape.POINT
    )

    position: Vector3 = field(
        default_factory=Vector3
    )

    size: Vector3 = field(
        default_factory=lambda: Vector3(
            1.0,
            1.0,
            1.0,
        )
    )

    emission_rate: float = 25.0

    max_particles: int = 250

    lifetime_min: float = 0.5
    lifetime_max: float = 1.5

    speed_min: float = 0.1
    speed_max: float = 0.8

    size_start_min: float = 1.0
    size_start_max: float = 2.0

    size_end_min: float = 0.0
    size_end_max: float = 0.5

    gravity: float = 0.0

    drag: float = 0.0

    spread: float = math.pi * 2.0

    direction: Vector3 = field(
        default_factory=lambda: Vector3(
            0.0,
            1.0,
            0.0,
        )
    )

    color_start: ParticleColor = field(
        default_factory=ParticleColor
    )

    color_end: ParticleColor = field(
        default_factory=lambda: ParticleColor(
            0.0,
            0.8,
            1.0,
            0.0,
        )
    )

    opacity_start: float = 1.0

    opacity_end: float = 0.0

    glow: float = 1.0

    rotation_speed_min: float = -1.0

    rotation_speed_max: float = 1.0

    blend_mode: ParticleBlendMode = (
        ParticleBlendMode.ADDITIVE
    )

    enabled: bool = True

    auto_emit: bool = True

    seed: Optional[int] = None


# ============================================================================
# PARTICLE EMITTER
# ============================================================================


@dataclass
class ParticleEmitter:
    id: str

    config: EmitterConfig

    particles: list[Particle] = field(
        default_factory=list
    )

    emission_accumulator: float = 0.0

    elapsed: float = 0.0

    state: ParticleState = (
        ParticleState.ACTIVE
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def alive_count(self) -> int:

        return sum(
            1
            for particle in self.particles
            if particle.alive
        )


# ============================================================================
# PARTICLE GROUP
# ============================================================================


@dataclass
class ParticleGroup:
    id: str

    emitter_ids: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# PARTICLE ENGINE
# ============================================================================


class ParticleEngine:
    """
    RENIX particle simulation engine.

    The engine is intentionally independent of the visual backend.

    Renderers can call:

        get_particles()

    or:

        get_snapshot()

    every frame.
    """

    def __init__(
        self,
        *,
        max_total_particles: int = 5000,
        target_fps: int = 60,
    ) -> None:

        self.max_total_particles = max(
            1,
            int(max_total_particles),
        )

        self.target_fps = max(
            1,
            int(target_fps),
        )

        self._emitters: dict[
            str,
            ParticleEmitter,
        ] = {}

        self._groups: dict[
            str,
            ParticleGroup,
        ] = {}

        self._callbacks: dict[
            str,
            list[Callable[..., None]],
        ] = {}

        self._lock = threading.RLock()

        self._running = False

        self._last_update = time.monotonic()

        self._time = 0.0

        self._random = random.Random()

        logger.debug(
            "Particle engine initialized"
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
            "Particle engine started"
        )

    def stop(
        self,
        *,
        clear: bool = False,
    ) -> None:

        with self._lock:

            self._running = False

            if clear:

                self._emitters.clear()

                self._groups.clear()

        self._emit(
            "stopped",
            self,
        )

    # ========================================================================
    # UPDATE
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

        self._last_update = now

        delta_time = max(
            0.0,
            min(
                0.1,
                float(delta_time),
            ),
        )

        if delta_time <= 0:
            return

        with self._lock:

            self._time += delta_time

            emitters = list(
                self._emitters.values()
            )

        for emitter in emitters:

            if (
                emitter.state
                != ParticleState.ACTIVE
            ):

                continue

            if emitter.config.enabled:

                self._update_emitter(
                    emitter,
                    delta_time,
                )

        self._enforce_global_limit()

        self._emit(
            "updated",
            delta_time,
        )

    # ========================================================================
    # EMITTER CREATION
    # ========================================================================

    def create_emitter(
        self,
        config: Optional[
            EmitterConfig
        ] = None,
        *,
        emitter_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> ParticleEmitter:

        if config is None:

            config = EmitterConfig()

        if name is not None:

            config.name = name

        if config.seed is not None:

            rng = random.Random(
                config.seed
            )

            # Mix the deterministic RNG into
            # the engine RNG.
            self._random.setstate(
                rng.getstate()
            )

        emitter = ParticleEmitter(
            id=(
                emitter_id
                or uuid.uuid4().hex
            ),
            config=config,
        )

        with self._lock:

            self._emitters[
                emitter.id
            ] = emitter

        self._emit(
            "emitter_created",
            emitter,
        )

        return emitter

    # ========================================================================
    # REMOVE EMITTER
    # ========================================================================

    def remove_emitter(
        self,
        emitter_id: str,
    ) -> bool:

        with self._lock:

            emitter = self._emitters.pop(
                emitter_id,
                None,
            )

        if emitter is None:
            return False

        with self._lock:

            for group in self._groups.values():

                if emitter_id in group.emitter_ids:

                    group.emitter_ids.remove(
                        emitter_id
                    )

        self._emit(
            "emitter_removed",
            emitter,
        )

        return True

    # ========================================================================
    # GET EMITTER
    # ========================================================================

    def get_emitter(
        self,
        emitter_id: str,
    ) -> Optional[ParticleEmitter]:

        with self._lock:

            return self._emitters.get(
                emitter_id
            )

    def get_emitters(
        self,
    ) -> list[ParticleEmitter]:

        with self._lock:

            return list(
                self._emitters.values()
            )

    # ========================================================================
    # EMISSION
    # ========================================================================

    def emit(
        self,
        emitter_id: str,
        count: int = 1,
        *,
        position: Optional[
            Vector3
        ] = None,
        velocity: Optional[
            Vector3
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> list[Particle]:

        emitter = self.get_emitter(
            emitter_id
        )

        if emitter is None:
            return []

        count = max(
            0,
            int(count),
        )

        particles: list[
            Particle
        ] = []

        for _ in range(count):

            if (
                emitter.alive_count()
                >= emitter.config.max_particles
            ):

                break

            if (
                self.total_particle_count()
                >= self.max_total_particles
            ):

                break

            particle = (
                self._spawn_particle(
                    emitter,
                    position=position,
                    velocity=velocity,
                    metadata=metadata,
                )
            )

            emitter.particles.append(
                particle
            )

            particles.append(
                particle
            )

            self._emit(
                "particle_spawned",
                particle,
                emitter,
            )

        return particles

    # ========================================================================
    # UPDATE EMITTER
    # ========================================================================

    def _update_emitter(
        self,
        emitter: ParticleEmitter,
        delta_time: float,
    ) -> None:

        emitter.elapsed += delta_time

        config = emitter.config

        # Automatic emission.
        if config.auto_emit:

            config_rate = max(
                0.0,
                config.emission_rate,
            )

            if config_rate > 0:

                emitter.emission_accumulator += (
                    config_rate
                    * delta_time
                )

                spawn_count = int(
                    emitter.emission_accumulator
                )

                if spawn_count > 0:

                    emitter.emission_accumulator -= (
                        spawn_count
                    )

                    self.emit(
                        emitter.id,
                        spawn_count,
                    )

        # Update particles.
        alive_particles: list[
            Particle
        ] = []

        for particle in emitter.particles:

            if not particle.alive:
                continue

            self._update_particle(
                particle,
                delta_time,
            )

            if particle.alive:

                alive_particles.append(
                    particle
                )

            else:

                self._emit(
                    "particle_died",
                    particle,
                    emitter,
                )

        emitter.particles = (
            alive_particles
        )

    # ========================================================================
    # UPDATE PARTICLE
    # ========================================================================

    def _update_particle(
        self,
        particle: Particle,
        delta_time: float,
    ) -> None:

        particle.age += delta_time

        if (
            particle.age
            >= particle.lifetime
        ):

            particle.state = (
                ParticleState.DEAD
            )

            particle.opacity = 0.0

            return

        # Acceleration.
        particle.velocity.x += (
            particle.acceleration.x
            * delta_time
        )

        particle.velocity.y += (
            particle.acceleration.y
            * delta_time
        )

        particle.velocity.z += (
            particle.acceleration.z
            * delta_time
        )

        # Gravity.
        particle.velocity.y -= (
            particle.gravity
            * delta_time
        )

        # Drag.
        if particle.drag > 0:

            drag_factor = max(
                0.0,
                1.0
                - particle.drag
                * delta_time,
            )

            particle.velocity.x *= (
                drag_factor
            )

            particle.velocity.y *= (
                drag_factor
            )

            particle.velocity.z *= (
                drag_factor
            )

        # Position.
        particle.position.x += (
            particle.velocity.x
            * delta_time
        )

        particle.position.y += (
            particle.velocity.y
            * delta_time
        )

        particle.position.z += (
            particle.velocity.z
            * delta_time
        )

        # Rotation.
        particle.rotation += (
            particle.rotation_speed
            * delta_time
        )

        # Normalize lifetime.
        progress = particle.progress

        # Size interpolation.
        particle.size = self._lerp(
            particle.start_size,
            particle.end_size,
            self._smoothstep(progress),
        )

        # Opacity interpolation.
        particle.opacity = self._lerp(
            particle.start_opacity,
            particle.end_opacity,
            progress,
        )

        # Color interpolation.
        particle.color = (
            self._lerp_color(
                particle.start_color,
                particle.end_color,
                progress,
            )
        )

        self._emit(
            "particle_updated",
            particle,
        )

    # ========================================================================
    # SPAWN PARTICLE
    # ========================================================================

    def _spawn_particle(
        self,
        emitter: ParticleEmitter,
        *,
        position: Optional[
            Vector3
        ] = None,
        velocity: Optional[
            Vector3
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Particle:

        config = emitter.config

        spawn_position = (
            position.copy()
            if position is not None
            else self._random_spawn_position(
                config
            )
        )

        if velocity is not None:

            particle_velocity = (
                velocity.copy()
            )

        else:

            particle_velocity = (
                self._random_velocity(
                    config
                )
            )

        lifetime = (
            self._random_range(
                config.lifetime_min,
                config.lifetime_max,
            )
        )

        start_size = (
            self._random_range(
                config.size_start_min,
                config.size_start_max,
            )
        )

        end_size = (
            self._random_range(
                config.size_end_min,
                config.size_end_max,
            )
        )

        rotation_speed = (
            self._random_range(
                config.rotation_speed_min,
                config.rotation_speed_max,
            )
        )

        start_color = (
            config.color_start.copy()
        )

        end_color = (
            config.color_end.copy()
        )

        particle = Particle(
            id=uuid.uuid4().hex,
            position=spawn_position,
            velocity=particle_velocity,
            acceleration=Vector3(),
            color=start_color.copy(),
            start_color=start_color,
            end_color=end_color,
            size=start_size,
            start_size=start_size,
            end_size=end_size,
            opacity=config.opacity_start,
            start_opacity=config.opacity_start,
            end_opacity=config.opacity_end,
            lifetime=max(
                0.01,
                lifetime,
            ),
            age=0.0,
            rotation=self._random_range(
                0.0,
                math.pi * 2.0,
            ),
            rotation_speed=rotation_speed,
            gravity=config.gravity,
            drag=config.drag,
            glow=config.glow,
            metadata=dict(
                metadata or {}
            ),
        )

        return particle

    # ========================================================================
    # RANDOM POSITION
    # ========================================================================

    def _random_spawn_position(
        self,
        config: EmitterConfig,
    ) -> Vector3:

        shape = config.shape

        if shape == EmitterShape.POINT:

            return config.position.copy()

        if shape == EmitterShape.BOX:

            return Vector3(
                config.position.x
                + self._random_range(
                    -config.size.x / 2,
                    config.size.x / 2,
                ),
                config.position.y
                + self._random_range(
                    -config.size.y / 2,
                    config.size.y / 2,
                ),
                config.position.z
                + self._random_range(
                    -config.size.z / 2,
                    config.size.z / 2,
                ),
            )

        if shape == EmitterShape.CIRCLE:

            angle = self._random_range(
                0.0,
                math.pi * 2.0,
            )

            radius = (
                math.sqrt(
                    self._random_range(
                        0.0,
                        1.0,
                    )
                )
                * config.size.x
            )

            return Vector3(
                config.position.x
                + math.cos(angle)
                * radius,
                config.position.y,
                config.position.z
                + math.sin(angle)
                * radius,
            )

        if shape == EmitterShape.RING:

            angle = self._random_range(
                0.0,
                math.pi * 2.0,
            )

            radius = max(
                0.0,
                config.size.x,
            )

            return Vector3(
                config.position.x
                + math.cos(angle)
                * radius,
                config.position.y,
                config.position.z
                + math.sin(angle)
                * radius,
            )

        if shape == EmitterShape.SPHERE:

            direction = self._random_unit_vector()

            radius = (
                self._random_range(
                    0.0,
                    1.0,
                )
                ** (1.0 / 3.0)
            ) * config.size.x

            return (
                config.position
                + direction * radius
            )

        if shape == EmitterShape.LINE:

            t = self._random_range(
                -0.5,
                0.5,
            )

            return Vector3(
                config.position.x
                + config.size.x * t,
                config.position.y
                + config.size.y * t,
                config.position.z
                + config.size.z * t,
            )

        return config.position.copy()

    # ========================================================================
    # RANDOM VELOCITY
    # ========================================================================

    def _random_velocity(
        self,
        config: EmitterConfig,
    ) -> Vector3:

        direction = (
            config.direction.normalized()
        )

        speed = self._random_range(
            config.speed_min,
            config.speed_max,
        )

        # Random spherical direction.
        random_direction = (
            self._random_unit_vector()
        )

        # Blend the requested direction
        # with random direction according
        # to spread.
        spread = max(
            0.0,
            min(
                math.pi,
                config.spread,
            ),
        )

        blend = spread / math.pi

        final_direction = Vector3(
            direction.x
            * (1.0 - blend)
            + random_direction.x
            * blend,

            direction.y
            * (1.0 - blend)
            + random_direction.y
            * blend,

            direction.z
            * (1.0 - blend)
            + random_direction.z
            * blend,
        ).normalized()

        return (
            final_direction * speed
        )

    # ========================================================================
    # BURST
    # ========================================================================

    def burst(
        self,
        position: Vector3,
        *,
        count: int = 50,
        color: Optional[
            ParticleColor
        ] = None,
        speed_min: float = 1.0,
        speed_max: float = 4.0,
        lifetime_min: float = 0.3,
        lifetime_max: float = 1.2,
        size_min: float = 1.0,
        size_max: float = 3.0,
        glow: float = 2.0,
    ) -> list[Particle]:

        config = EmitterConfig(
            name="burst",
            shape=EmitterShape.POINT,
            position=position.copy(),
            emission_rate=0.0,
            max_particles=count,
            lifetime_min=lifetime_min,
            lifetime_max=lifetime_max,
            speed_min=speed_min,
            speed_max=speed_max,
            size_start_min=size_min,
            size_start_max=size_max,
            size_end_min=0.0,
            size_end_max=0.5,
            color_start=(
                color.copy()
                if color
                else ParticleColor(
                    0.0,
                    1.0,
                    0.7,
                    1.0,
                )
            ),
            color_end=(
                ParticleColor(
                    0.0,
                    0.8,
                    1.0,
                    0.0,
                )
            ),
            opacity_start=1.0,
            opacity_end=0.0,
            glow=glow,
            auto_emit=False,
            spread=math.pi * 2,
        )

        emitter = self.create_emitter(
            config
        )

        particles = self.emit(
            emitter.id,
            count,
        )

        # Burst emitters can be removed
        # after their particles die.
        emitter.metadata[
            "temporary"
        ] = True

        return particles

    # ========================================================================
    # TRAIL
    # ========================================================================

    def trail(
        self,
        position: Vector3,
        *,
        direction: Optional[
            Vector3
        ] = None,
        count: int = 5,
        color: Optional[
            ParticleColor
        ] = None,
        size: float = 2.0,
        lifetime: float = 0.5,
    ) -> list[Particle]:

        direction = (
            direction.copy()
            if direction
            else Vector3()
        )

        config = EmitterConfig(
            name="trail",
            shape=EmitterShape.POINT,
            position=position.copy(),
            emission_rate=0.0,
            max_particles=count,
            lifetime_min=lifetime,
            lifetime_max=lifetime,
            speed_min=0.05,
            speed_max=0.2,
            size_start_min=size,
            size_start_max=size,
            size_end_min=0.0,
            size_end_max=0.0,
            color_start=(
                color.copy()
                if color
                else ParticleColor()
            ),
            color_end=(
                ParticleColor(
                    0.0,
                    1.0,
                    0.8,
                    0.0,
                )
            ),
            opacity_start=0.8,
            opacity_end=0.0,
            gravity=0.0,
            drag=1.0,
            spread=0.2,
            direction=direction,
            auto_emit=False,
        )

        emitter = self.create_emitter(
            config
        )

        particles = self.emit(
            emitter.id,
            count,
        )

        emitter.metadata[
            "temporary"
        ] = True

        return particles

    # ========================================================================
    # GESTURE EFFECT
    # ========================================================================

    def gesture_burst(
        self,
        position: Vector3,
        *,
        gesture: str = "pinch",
    ) -> list[Particle]:

        gesture = gesture.lower()

        presets = {

            "pinch": {
                "count": 20,
                "speed_min": 0.4,
                "speed_max": 1.8,
                "size_min": 1.0,
                "size_max": 2.0,
                "glow": 2.5,
            },

            "grab": {
                "count": 35,
                "speed_min": 0.5,
                "speed_max": 2.5,
                "size_min": 1.0,
                "size_max": 2.5,
                "glow": 3.0,
            },

            "swipe": {
                "count": 50,
                "speed_min": 1.0,
                "speed_max": 4.0,
                "size_min": 0.8,
                "size_max": 2.0,
                "glow": 2.0,
            },

            "point": {
                "count": 15,
                "speed_min": 0.2,
                "speed_max": 1.0,
                "size_min": 0.8,
                "size_max": 1.8,
                "glow": 2.0,
            },

            "palm": {
                "count": 40,
                "speed_min": 0.3,
                "speed_max": 1.5,
                "size_min": 1.0,
                "size_max": 2.5,
                "glow": 2.5,
            },
        }

        preset = presets.get(
            gesture,
            presets["pinch"],
        )

        return self.burst(
            position,
            **preset,
        )

    # ========================================================================
    # PARTICLE LIMIT
    # ========================================================================

    def _enforce_global_limit(
        self,
    ) -> None:

        total = (
            self.total_particle_count()
        )

        if total <= self.max_total_particles:

            self._cleanup_temporary_emitters()

            return

        excess = (
            total
            - self.max_total_particles
        )

        with self._lock:

            emitters = list(
                self._emitters.values()
            )

        # Remove oldest particles first.
        candidates: list[
            tuple[
                float,
                ParticleEmitter,
                Particle,
            ]
        ] = []

        for emitter in emitters:

            for particle in emitter.particles:

                candidates.append(
                    (
                        particle.age,
                        emitter,
                        particle,
                    )
                )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        for _, emitter, particle in (
            candidates[:excess]
        ):

            try:

                emitter.particles.remove(
                    particle
                )

            except ValueError:

                pass

        self._cleanup_temporary_emitters()

    def _cleanup_temporary_emitters(
        self,
    ) -> None:

        with self._lock:

            emitters = list(
                self._emitters.values()
            )

        for emitter in emitters:

            if not emitter.metadata.get(
                "temporary",
                False,
            ):

                continue

            if emitter.particles:
                continue

            self.remove_emitter(
                emitter.id
            )

    # ========================================================================
    # GROUPS
    # ========================================================================

    def create_group(
        self,
        emitter_ids: Optional[
            list[str]
        ] = None,
        *,
        group_id: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> ParticleGroup:

        group = ParticleGroup(
            id=(
                group_id
                or uuid.uuid4().hex
            ),
            emitter_ids=list(
                emitter_ids or []
            ),
            metadata=dict(
                metadata or {}
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
        emitter_id: str,
    ) -> bool:

        with self._lock:

            group = self._groups.get(
                group_id
            )

            emitter = self._emitters.get(
                emitter_id
            )

            if (
                group is None
                or emitter is None
            ):

                return False

            if (
                emitter_id
                not in group.emitter_ids
            ):

                group.emitter_ids.append(
                    emitter_id
                )

        return True

    def clear_group(
        self,
        group_id: str,
    ) -> bool:

        with self._lock:

            group = self._groups.get(
                group_id
            )

            if group is None:
                return False

            group.emitter_ids.clear()

        return True

    def remove_group(
        self,
        group_id: str,
    ) -> bool:

        with self._lock:

            return (
                self._groups.pop(
                    group_id,
                    None,
                )
                is not None
            )

    # ========================================================================
    # PARTICLE ACCESS
    # ========================================================================

    def get_particles(
        self,
        *,
        emitter_id: Optional[str] = None,
    ) -> list[Particle]:

        with self._lock:

            if emitter_id is not None:

                emitter = (
                    self._emitters.get(
                        emitter_id
                    )
                )

                if emitter is None:
                    return []

                return [
                    particle
                    for particle
                    in emitter.particles
                    if particle.alive
                ]

            particles: list[
                Particle
            ] = []

            for emitter in (
                self._emitters.values()
            ):

                particles.extend(
                    particle
                    for particle
                    in emitter.particles
                    if particle.alive
                )

            return particles

    def get_snapshot(
        self,
    ) -> list[dict[str, Any]]:

        particles = self.get_particles()

        return [
            {
                "id": particle.id,

                "position": {
                    "x": particle.position.x,
                    "y": particle.position.y,
                    "z": particle.position.z,
                },

                "velocity": {
                    "x": particle.velocity.x,
                    "y": particle.velocity.y,
                    "z": particle.velocity.z,
                },

                "color": {
                    "r": particle.color.r,
                    "g": particle.color.g,
                    "b": particle.color.b,
                    "a": particle.color.a,
                },

                "size": particle.size,

                "opacity": particle.opacity,

                "rotation": particle.rotation,

                "glow": particle.glow,

                "progress": particle.progress,

                "metadata": dict(
                    particle.metadata
                ),
            }
            for particle in particles
        ]

    def total_particle_count(
        self,
    ) -> int:

        with self._lock:

            return sum(
                emitter.alive_count()
                for emitter
                in self._emitters.values()
            )

    # ========================================================================
    # GLOBAL CONTROLS
    # ========================================================================

    def pause_all(self) -> None:

        with self._lock:

            for emitter in (
                self._emitters.values()
            ):

                emitter.state = (
                    ParticleState.PAUSED
                )

                for particle in (
                    emitter.particles
                ):

                    if particle.alive:

                        particle.state = (
                            ParticleState.PAUSED
                        )

    def resume_all(self) -> None:

        with self._lock:

            for emitter in (
                self._emitters.values()
            ):

                emitter.state = (
                    ParticleState.ACTIVE
                )

                for particle in (
                    emitter.particles
                ):

                    if (
                        particle.state
                        == ParticleState.PAUSED
                    ):

                        particle.state = (
                            ParticleState.ACTIVE
                        )

    def clear(self) -> None:

        with self._lock:

            self._emitters.clear()

            self._groups.clear()

        self._emit(
            "cleared"
        )

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _random_range(
        self,
        minimum: float,
        maximum: float,
    ) -> float:

        return self._random.uniform(
            float(minimum),
            float(maximum),
        )

    def _random_unit_vector(
        self,
    ) -> Vector3:

        z = self._random_range(
            -1.0,
            1.0,
        )

        angle = self._random_range(
            0.0,
            math.pi * 2.0,
        )

        radius = math.sqrt(
            max(
                0.0,
                1.0 - z * z,
            )
        )

        return Vector3(
            radius * math.cos(angle),
            radius * math.sin(angle),
            z,
        )

    @staticmethod
    def _lerp(
        a: float,
        b: float,
        t: float,
    ) -> float:

        return (
            a
            + (b - a) * t
        )

    @staticmethod
    def _smoothstep(
        t: float,
    ) -> float:

        t = max(
            0.0,
            min(
                1.0,
                t,
            ),
        )

        return (
            t * t * (3.0 - 2.0 * t)
        )

    @staticmethod
    def _lerp_color(
        a: ParticleColor,
        b: ParticleColor,
        t: float,
    ) -> ParticleColor:

        return ParticleColor(
            r=ParticleEngine._lerp(
                a.r,
                b.r,
                t,
            ),
            g=ParticleEngine._lerp(
                a.g,
                b.g,
                t,
            ),
            b=ParticleEngine._lerp(
                a.b,
                b.b,
                t,
            ),
            a=ParticleEngine._lerp(
                a.a,
                b.a,
                t,
            ),
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
                    "Particle event callback failed"
                )

    # ========================================================================
    # STATUS
    # ========================================================================

    @property
    def running(self) -> bool:

        with self._lock:

            return self._running

    def status(self) -> dict[str, Any]:

        with self._lock:

            return {
                "running": self._running,
                "emitters": len(
                    self._emitters
                ),
                "groups": len(
                    self._groups
                ),
                "particles": (
                    self.total_particle_count()
                ),
                "max_particles": (
                    self.max_total_particles
                ),
                "utilization": (
                    self.total_particle_count()
                    / self.max_total_particles
                ),
            }


# ============================================================================
# PRESET FACTORIES
# ============================================================================


def create_ambient_emitter(
    engine: ParticleEngine,
    *,
    position: Optional[
        Vector3
    ] = None,
) -> ParticleEmitter:

    config = EmitterConfig(
        name="ambient_hologram",
        shape=EmitterShape.BOX,
        position=(
            position.copy()
            if position
            else Vector3()
        ),
        size=Vector3(
            12.0,
            8.0,
            6.0,
        ),
        emission_rate=18.0,
        max_particles=400,
        lifetime_min=3.0,
        lifetime_max=7.0,
        speed_min=0.02,
        speed_max=0.12,
        size_start_min=0.5,
        size_start_max=1.5,
        size_end_min=0.0,
        size_end_max=0.4,
        gravity=-0.01,
        drag=0.05,
        opacity_start=0.25,
        opacity_end=0.0,
        glow=1.5,
        spread=math.pi * 2,
    )

    return engine.create_emitter(
        config
    )


def create_energy_emitter(
    engine: ParticleEngine,
    position: Vector3,
) -> ParticleEmitter:

    config = EmitterConfig(
        name="energy_core",
        shape=EmitterShape.SPHERE,
        position=position.copy(),
        size=Vector3(
            1.5,
            1.5,
            1.5,
        ),
        emission_rate=45.0,
        max_particles=500,
        lifetime_min=0.4,
        lifetime_max=1.2,
        speed_min=0.3,
        speed_max=1.8,
        size_start_min=1.0,
        size_start_max=2.5,
        size_end_min=0.0,
        size_end_max=0.5,
        opacity_start=0.9,
        opacity_end=0.0,
        glow=3.0,
        spread=math.pi,
    )

    return engine.create_emitter(
        config
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "ParticleState",
    "EmitterShape",
    "ParticleBlendMode",
    "Vector3",
    "ParticleColor",
    "Particle",
    "EmitterConfig",
    "ParticleEmitter",
    "ParticleGroup",
    "ParticleEngine",
    "create_ambient_emitter",
    "create_energy_emitter",
]


