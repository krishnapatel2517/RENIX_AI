"""
RENIX Holographic UI
Settings UI
==============

Central settings/state layer for the RENIX holographic interface.

Responsibilities:
- UI preferences
- Theme selection
- HUD settings
- Animation settings
- Hologram settings
- Gesture settings
- Voice UI settings
- Performance settings
- Accessibility settings
- Persistence-ready configuration
- Event-driven updates

This module does not directly render graphics.
Rendering is handled by the holographic UI engine.
"""

from __future__ import annotations

import copy
import threading
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ============================================================================
# ENUMS
# ============================================================================


class Theme(str, Enum):
    RENIX = "renix"
    HUD = "hud"
    CUSTOM = "custom"


class AnimationQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class GraphicsQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class InteractionMode(str, Enum):
    VOICE = "voice"
    GESTURE = "gesture"
    HYBRID = "hybrid"


class AccentStyle(str, Enum):
    GREEN = "green"
    CYAN = "cyan"
    BLUE = "blue"
    PURPLE = "purple"
    RED = "red"
    CUSTOM = "custom"


class SettingsSection(str, Enum):
    GENERAL = "general"
    APPEARANCE = "appearance"
    HOLOGRAM = "hologram"
    HUD = "hud"
    ANIMATION = "animation"
    GESTURES = "gestures"
    VOICE = "voice"
    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"
    SECURITY = "security"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class AppearanceSettings:

    theme: Theme = Theme.RENIX

    accent_style: AccentStyle = (
        AccentStyle.GREEN
    )

    custom_accent: str = "#00FF88"

    opacity: float = 0.96

    blur: float = 0.75

    glow: float = 0.90

    scanlines: bool = True

    glass_effect: bool = True

    dark_mode: bool = True


@dataclass
class HologramSettings:

    enabled: bool = True

    intensity: float = 1.0

    brightness: float = 1.0

    glow_strength: float = 0.85

    transparency: float = 0.15

    particle_density: float = 1.0

    hologram_scale: float = 1.0

    rotation_speed: float = 1.0

    floating_motion: bool = True

    grid_enabled: bool = True

    scan_effect: bool = True


@dataclass
class HUDSettings:

    enabled: bool = True

    show_clock: bool = True

    show_system_status: bool = True

    show_cpu: bool = True

    show_gpu: bool = True

    show_ram: bool = True

    show_network: bool = True

    show_weather: bool = True

    show_calendar: bool = True

    show_tasks: bool = True

    show_media: bool = True

    compact_mode: bool = False

    transparency: float = 0.88


@dataclass
class AnimationSettings:

    enabled: bool = True

    quality: AnimationQuality = (
        AnimationQuality.HIGH
    )

    speed: float = 1.0

    transitions: bool = True

    particles: bool = True

    ripple_effects: bool = True

    hologram_flicker: bool = True

    gesture_trails: bool = True

    smooth_cursor: bool = True


@dataclass
class GestureSettings:

    enabled: bool = True

    sensitivity: float = 1.0

    smoothing: float = 0.70

    pinch_enabled: bool = True

    swipe_enabled: bool = True

    grab_enabled: bool = True

    rotate_enabled: bool = True

    zoom_enabled: bool = True

    drag_enabled: bool = True

    point_enabled: bool = True

    palm_enabled: bool = True

    two_hand_enabled: bool = True

    gesture_cursor: bool = True


@dataclass
class VoiceSettings:

    enabled: bool = True

    microphone_enabled: bool = True

    voice_feedback: bool = True

    show_transcript: bool = True

    wake_word_indicator: bool = True

    conversation_mode: bool = True

    interruption_enabled: bool = True

    visual_waveform: bool = True

    waveform_intensity: float = 1.0


@dataclass
class PerformanceSettings:

    graphics_quality: GraphicsQuality = (
        GraphicsQuality.HIGH
    )

    max_fps: int = 60

    particle_limit: int = 5000

    enable_gpu_acceleration: bool = True

    adaptive_quality: bool = True

    reduce_effects_on_load: bool = True

    low_power_mode: bool = False

    show_fps: bool = False

    show_debug_info: bool = False


@dataclass
class AccessibilitySettings:

    large_text: bool = False

    high_contrast: bool = False

    reduce_motion: bool = False

    reduce_transparency: bool = False

    screen_reader_support: bool = False

    keyboard_navigation: bool = True

    voice_navigation: bool = True


@dataclass
class GeneralSettings:

    startup_enabled: bool = True

    minimize_to_tray: bool = True

    show_welcome: bool = True

    confirmation_for_dangerous_actions: bool = True

    interaction_mode: InteractionMode = (
        InteractionMode.HYBRID
    )

    language: str = "en"

    notifications_enabled: bool = True


@dataclass
class SettingsState:

    general: GeneralSettings = field(
        default_factory=GeneralSettings
    )

    appearance: AppearanceSettings = field(
        default_factory=AppearanceSettings
    )

    hologram: HologramSettings = field(
        default_factory=HologramSettings
    )

    hud: HUDSettings = field(
        default_factory=HUDSettings
    )

    animation: AnimationSettings = field(
        default_factory=AnimationSettings
    )

    gestures: GestureSettings = field(
        default_factory=GestureSettings
    )

    voice: VoiceSettings = field(
        default_factory=VoiceSettings
    )

    performance: PerformanceSettings = field(
        default_factory=PerformanceSettings
    )

    accessibility: AccessibilitySettings = field(
        default_factory=AccessibilitySettings
    )

    custom: dict[str, Any] = field(
        default_factory=dict
    )

    last_updated: float = field(
        default_factory=time.time
    )


# ============================================================================
# SETTINGS UI
# ============================================================================


class SettingsUI:

    def __init__(
        self,
        *,
        initial_settings: Optional[
            SettingsState
        ] = None,
    ) -> None:

        self._lock = threading.RLock()

        self.settings = (
            copy.deepcopy(
                initial_settings
            )
            if initial_settings is not None
            else SettingsState()
        )

        self._active_section = (
            SettingsSection.GENERAL
        )

        self._callbacks: dict[
            str,
            list[Callable[..., None]],
        ] = {}

        self._dirty = False

        self._visible = False

        self._running = False

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self) -> None:

        with self._lock:
            self._running = True

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

    def show(self) -> None:

        with self._lock:
            self._visible = True

        self._emit(
            "visibility_changed",
            True,
        )

    def hide(self) -> None:

        with self._lock:
            self._visible = False

        self._emit(
            "visibility_changed",
            False,
        )

    def toggle(self) -> bool:

        with self._lock:
            self._visible = not self._visible
            value = self._visible

        self._emit(
            "visibility_changed",
            value,
        )

        return value

    @property
    def visible(self) -> bool:

        with self._lock:
            return self._visible

    # ========================================================================
    # SECTIONS
    # ========================================================================

    @property
    def active_section(
        self,
    ) -> SettingsSection:

        with self._lock:
            return self._active_section

    def select_section(
        self,
        section: SettingsSection | str,
    ) -> SettingsSection:

        if isinstance(section, str):
            section = SettingsSection(
                section.lower()
            )

        with self._lock:
            self._active_section = section

        self._emit(
            "section_changed",
            section,
        )

        return section

    # ========================================================================
    # GENERIC GET / SET
    # ========================================================================

    def get(
        self,
        section: str,
        key: str,
        default: Any = None,
    ) -> Any:

        with self._lock:

            target = getattr(
                self.settings,
                section,
                None,
            )

            if target is None:
                return default

            return getattr(
                target,
                key,
                default,
            )

    def set(
        self,
        section: str,
        key: str,
        value: Any,
    ) -> bool:

        with self._lock:

            target = getattr(
                self.settings,
                section,
                None,
            )

            if target is None:
                return False

            if not hasattr(
                target,
                key,
            ):

                return False

            old_value = getattr(
                target,
                key,
            )

            value = self._normalize_value(
                old_value,
                value,
            )

            setattr(
                target,
                key,
                value,
            )

            self.settings.last_updated = (
                time.time()
            )

            self._dirty = True

        self._emit(
            "setting_changed",
            section,
            key,
            old_value,
            value,
        )

        return True

    @staticmethod
    def _normalize_value(
        old_value: Any,
        value: Any,
    ) -> Any:

        if isinstance(
            old_value,
            Enum,
        ):

            enum_type = type(old_value)

            if isinstance(
                value,
                enum_type,
            ):
                return value

            return enum_type(
                str(value).lower()
            )

        if isinstance(
            old_value,
            bool,
        ):
            return bool(value)

        if isinstance(
            old_value,
            int,
        ):

            return int(value)

        if isinstance(
            old_value,
            float,
        ):

            return float(value)

        return value

    # ========================================================================
    # APPEARANCE
    # ========================================================================

    def set_theme(
        self,
        theme: Theme | str,
    ) -> bool:

        return self.set(
            "appearance",
            "theme",
            theme,
        )

    def set_accent(
        self,
        accent: AccentStyle | str,
        custom_hex: Optional[str] = None,
    ) -> bool:

        success = self.set(
            "appearance",
            "accent_style",
            accent,
        )

        if (
            success
            and custom_hex is not None
        ):

            self.set(
                "appearance",
                "custom_accent",
                custom_hex,
            )

        return success

    def set_opacity(
        self,
        value: float,
    ) -> bool:

        value = max(
            0.1,
            min(1.0, float(value)),
        )

        return self.set(
            "appearance",
            "opacity",
            value,
        )

    def set_glow(
        self,
        value: float,
    ) -> bool:

        value = max(
            0.0,
            min(1.0, float(value)),
        )

        return self.set(
            "appearance",
            "glow",
            value,
        )

    # ========================================================================
    # HOLOGRAM
    # ========================================================================

    def set_hologram_enabled(
        self,
        enabled: bool,
    ) -> bool:

        return self.set(
            "hologram",
            "enabled",
            enabled,
        )

    def set_hologram_intensity(
        self,
        value: float,
    ) -> bool:

        value = max(
            0.0,
            min(2.0, float(value)),
        )

        return self.set(
            "hologram",
            "intensity",
            value,
        )

    def set_particle_density(
        self,
        value: float,
    ) -> bool:

        value = max(
            0.0,
            min(3.0, float(value)),
        )

        return self.set(
            "hologram",
            "particle_density",
            value,
        )

    # ========================================================================
    # HUD
    # ========================================================================

    def toggle_hud(
        self,
        enabled: Optional[bool] = None,
    ) -> bool:

        if enabled is None:

            enabled = not self.get(
                "hud",
                "enabled",
                True,
            )

        self.set(
            "hud",
            "enabled",
            enabled,
        )

        return bool(enabled)

    def toggle_widget(
        self,
        widget: str,
        enabled: Optional[bool] = None,
    ) -> bool:

        key = f"show_{widget}"

        current = self.get(
            "hud",
            key,
            False,
        )

        if enabled is None:
            enabled = not current

        self.set(
            "hud",
            key,
            enabled,
        )

        return bool(enabled)

    # ========================================================================
    # ANIMATION
    # ========================================================================

    def set_animation_quality(
        self,
        quality: AnimationQuality | str,
    ) -> bool:

        return self.set(
            "animation",
            "quality",
            quality,
        )

    def set_animation_speed(
        self,
        speed: float,
    ) -> bool:

        speed = max(
            0.0,
            min(3.0, float(speed)),
        )

        return self.set(
            "animation",
            "speed",
            speed,
        )

    # ========================================================================
    # GESTURES
    # ========================================================================

    def set_gesture_sensitivity(
        self,
        sensitivity: float,
    ) -> bool:

        sensitivity = max(
            0.1,
            min(3.0, float(sensitivity)),
        )

        return self.set(
            "gestures",
            "sensitivity",
            sensitivity,
        )

    def set_gesture_smoothing(
        self,
        smoothing: float,
    ) -> bool:

        smoothing = max(
            0.0,
            min(1.0, float(smoothing)),
        )

        return self.set(
            "gestures",
            "smoothing",
            smoothing,
        )

    def toggle_gesture(
        self,
        gesture: str,
        enabled: Optional[bool] = None,
    ) -> bool:

        key = f"{gesture}_enabled"

        current = self.get(
            "gestures",
            key,
            False,
        )

        if enabled is None:
            enabled = not current

        self.set(
            "gestures",
            key,
            enabled,
        )

        return bool(enabled)

    # ========================================================================
    # VOICE
    # ========================================================================

    def toggle_voice(
        self,
        enabled: Optional[bool] = None,
    ) -> bool:

        current = self.get(
            "voice",
            "enabled",
            True,
        )

        if enabled is None:
            enabled = not current

        self.set(
            "voice",
            "enabled",
            enabled,
        )

        return bool(enabled)

    def set_waveform_intensity(
        self,
        value: float,
    ) -> bool:

        value = max(
            0.0,
            min(3.0, float(value)),
        )

        return self.set(
            "voice",
            "waveform_intensity",
            value,
        )

    # ========================================================================
    # PERFORMANCE
    # ========================================================================

    def set_graphics_quality(
        self,
        quality: GraphicsQuality | str,
    ) -> bool:

        return self.set(
            "performance",
            "graphics_quality",
            quality,
        )

    def set_fps(
        self,
        fps: int,
    ) -> bool:

        fps = max(
            15,
            min(240, int(fps)),
        )

        return self.set(
            "performance",
            "max_fps",
            fps,
        )

    def set_particle_limit(
        self,
        limit: int,
    ) -> bool:

        limit = max(
            100,
            min(100000, int(limit)),
        )

        return self.set(
            "performance",
            "particle_limit",
            limit,
        )

    def set_low_power_mode(
        self,
        enabled: bool,
    ) -> bool:

        success = self.set(
            "performance",
            "low_power_mode",
            enabled,
        )

        if success and enabled:

            self.set(
                "performance",
                "adaptive_quality",
                True,
            )

            self.set(
                "performance",
                "reduce_effects_on_load",
                True,
            )

        return success

    # ========================================================================
    # ACCESSIBILITY
    # ========================================================================

    def toggle_reduce_motion(
        self,
        enabled: Optional[bool] = None,
    ) -> bool:

        current = self.get(
            "accessibility",
            "reduce_motion",
            False,
        )

        if enabled is None:
            enabled = not current

        self.set(
            "accessibility",
            "reduce_motion",
            enabled,
        )

        if enabled:

            self.set(
                "animation",
                "speed",
                0.5,
            )

            self.set(
                "animation",
                "hologram_flicker",
                False,
            )

        return bool(enabled)

    def set_large_text(
        self,
        enabled: bool,
    ) -> bool:

        return self.set(
            "accessibility",
            "large_text",
            enabled,
        )

    def set_high_contrast(
        self,
        enabled: bool,
    ) -> bool:

        return self.set(
            "accessibility",
            "high_contrast",
            enabled,
        )

    # ========================================================================
    # PRESETS
    # ========================================================================

    def apply_preset(
        self,
        preset: str,
    ) -> bool:

        preset = preset.lower().strip()

        if preset == "performance":

            self.set(
                "performance",
                "graphics_quality",
                GraphicsQuality.LOW,
            )

            self.set(
                "performance",
                "max_fps",
                30,
            )

            self.set(
                "performance",
                "particle_limit",
                1000,
            )

            self.set(
                "hologram",
                "particle_density",
                0.4,
            )

            self.set(
                "animation",
                "quality",
                AnimationQuality.LOW,
            )

            return True

        if preset == "balanced":

            self.set(
                "performance",
                "graphics_quality",
                GraphicsQuality.MEDIUM,
            )

            self.set(
                "performance",
                "max_fps",
                60,
            )

            self.set(
                "performance",
                "particle_limit",
                3000,
            )

            self.set(
                "hologram",
                "particle_density",
                0.8,
            )

            self.set(
                "animation",
                "quality",
                AnimationQuality.MEDIUM,
            )

            return True

        if preset == "cinematic":

            self.set(
                "performance",
                "graphics_quality",
                GraphicsQuality.ULTRA,
            )

            self.set(
                "performance",
                "max_fps",
                120,
            )

            self.set(
                "performance",
                "particle_limit",
                15000,
            )

            self.set(
                "hologram",
                "particle_density",
                2.0,
            )

            self.set(
                "hologram",
                "glow_strength",
                1.0,
            )

            self.set(
                "animation",
                "quality",
                AnimationQuality.ULTRA,
            )

            return True

        if preset == "minimal":

            self.set(
                "hologram",
                "particle_density",
                0.2,
            )

            self.set(
                "hologram",
                "grid_enabled",
                False,
            )

            self.set(
                "hologram",
                "scan_effect",
                False,
            )

            self.set(
                "animation",
                "particles",
                False,
            )

            self.set(
                "animation",
                "ripple_effects",
                False,
            )

            self.set(
                "hud",
                "compact_mode",
                True,
            )

            return True

        if preset == "jarvis":

            self.set(
                "appearance",
                "accent_style",
                AccentStyle.CYAN,
            )

            self.set(
                "appearance",
                "custom_accent",
                "#00D9FF",
            )

            self.set(
                "hologram",
                "enabled",
                True,
            )

            self.set(
                "hologram",
                "glow_strength",
                0.95,
            )

            self.set(
                "animation",
                "quality",
                AnimationQuality.HIGH,
            )

            self.set(
                "gestures",
                "enabled",
                True,
            )

            self.set(
                "voice",
                "enabled",
                True,
            )

            return True

        if preset == "renix":

            self.set(
                "appearance",
                "theme",
                Theme.RENIX,
            )

            self.set(
                "appearance",
                "accent_style",
                AccentStyle.GREEN,
            )

            self.set(
                "appearance",
                "custom_accent",
                "#00FF88",
            )

            self.set(
                "hologram",
                "enabled",
                True,
            )

            self.set(
                "hologram",
                "glow_strength",
                0.90,
            )

            self.set(
                "animation",
                "quality",
                AnimationQuality.HIGH,
            )

            self.set(
                "gestures",
                "enabled",
                True,
            )

            self.set(
                "voice",
                "enabled",
                True,
            )

            self.set(
                "general",
                "interaction_mode",
                InteractionMode.HYBRID,
            )

            return True

        return False

    # ========================================================================
    # RESET
    # ========================================================================

    def reset_section(
        self,
        section: SettingsSection | str,
    ) -> bool:

        if isinstance(section, str):

            try:

                section = SettingsSection(
                    section.lower()
                )

            except ValueError:

                return False

        mapping = {
            SettingsSection.GENERAL:
                GeneralSettings,

            SettingsSection.APPEARANCE:
                AppearanceSettings,

            SettingsSection.HOLOGRAM:
                HologramSettings,

            SettingsSection.HUD:
                HUDSettings,

            SettingsSection.ANIMATION:
                AnimationSettings,

            SettingsSection.GESTURES:
                GestureSettings,

            SettingsSection.VOICE:
                VoiceSettings,

            SettingsSection.PERFORMANCE:
                PerformanceSettings,

            SettingsSection.ACCESSIBILITY:
                AccessibilitySettings,
        }

        cls = mapping.get(section)

        if cls is None:
            return False

        with self._lock:

            setattr(
                self.settings,
                section.value,
                cls(),
            )

            self.settings.last_updated = (
                time.time()
            )

            self._dirty = True

        self._emit(
            "section_reset",
            section,
        )

        return True

    def reset_all(self) -> None:

        with self._lock:

            self.settings = SettingsState()

            self._dirty = True

        self._emit(
            "settings_reset",
            self.settings,
        )

    # ========================================================================
    # IMPORT / EXPORT
    # ========================================================================

    def to_dict(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return self._serialize(
                self.settings
            )

    def _serialize(
        self,
        value: Any,
    ) -> Any:

        if isinstance(
            value,
            Enum,
        ):

            return value.value

        if hasattr(
            value,
            "__dataclass_fields__",
        ):

            result = {}

            for key in (
                value.__dataclass_fields__
            ):

                result[key] = self._serialize(
                    getattr(
                        value,
                        key,
                    )
                )

            return result

        if isinstance(
            value,
            dict,
        ):

            return {
                key: self._serialize(
                    item
                )
                for key, item
                in value.items()
            }

        if isinstance(
            value,
            list,
        ):

            return [
                self._serialize(item)
                for item in value
            ]

        return value

    def export_settings(
        self,
    ) -> dict[str, Any]:

        return copy.deepcopy(
            self.to_dict()
        )

    # ========================================================================
    # STATUS
    # ========================================================================

    @property
    def dirty(self) -> bool:

        with self._lock:
            return self._dirty

    def mark_saved(self) -> None:

        with self._lock:
            self._dirty = False

        self._emit(
            "saved",
            self.settings,
        )

    def get_status(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "running": self._running,
                "visible": self._visible,
                "active_section": (
                    self._active_section.value
                ),
                "dirty": self._dirty,
                "theme": (
                    self.settings
                    .appearance
                    .theme
                    .value
                ),
                "accent": (
                    self.settings
                    .appearance
                    .accent_style
                    .value
                ),
                "hologram": (
                    self.settings
                    .hologram
                    .enabled
                ),
                "hud": (
                    self.settings
                    .hud
                    .enabled
                ),
                "gestures": (
                    self.settings
                    .gestures
                    .enabled
                ),
                "voice": (
                    self.settings
                    .voice
                    .enabled
                ),
                "graphics_quality": (
                    self.settings
                    .performance
                    .graphics_quality
                    .value
                ),
                "fps": (
                    self.settings
                    .performance
                    .max_fps
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

                # UI callbacks must never
                # crash RENIX.
                pass


# ============================================================================
# FACTORY
# ============================================================================


def create_settings_ui() -> SettingsUI:

    ui = SettingsUI()

    ui.start()

    return ui


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "Theme",
    "AnimationQuality",
    "GraphicsQuality",
    "InteractionMode",
    "AccentStyle",
    "SettingsSection",
    "AppearanceSettings",
    "HologramSettings",
    "HUDSettings",
    "AnimationSettings",
    "GestureSettings",
    "VoiceSettings",
    "PerformanceSettings",
    "AccessibilitySettings",
    "GeneralSettings",
    "SettingsState",
    "SettingsUI",
    "create_settings_ui",
]


