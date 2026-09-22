"""
RENIX Configuration Manager
===========================

Central configuration system for RENIX.

Responsibilities:
    - Load config.yaml
    - Load .env
    - Provide default configuration
    - Support nested configuration access
    - Support environment variable overrides
    - Validate basic configuration values
    - Save configuration changes
    - Provide typed helper methods
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigManager:
    """
    Central configuration manager for RENIX.

    Configuration sources, in priority order:

        1. Built-in defaults
        2. config.yaml
        3. .env
        4. Environment variables

    Example:

        config.get("ai.provider")

        config.get("voice.enabled")

        config.set("ui.theme", "holographic")
    """

    # ------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------

    def __init__(
        self,
        root_dir: Path,
    ) -> None:

        self.root_dir = Path(
            root_dir
        ).resolve()

        self.config_file = (
            self.root_dir / "config.yaml"
        )

        self.env_file = (
            self.root_dir / ".env"
        )

        self.logger = logging.getLogger(
            "RENIX.ConfigManager"
        )

        self._config: dict[str, Any] = {}

        self._defaults = (
            self._build_default_config()
        )

        self._initialized = False

    # ------------------------------------------------------------------
    # DEFAULT CONFIGURATION
    # ------------------------------------------------------------------

    @staticmethod
    def _build_default_config() -> dict[str, Any]:
        """
        Build the complete built-in RENIX configuration.

        These defaults ensure RENIX can still start even if
        config.yaml does not exist.
        """

        return {
            "renix": {
                "name": "RENIX",
                "version": "1.0.0",
                "environment": "development",
                "debug": False,
                "first_run": True,
            },

            "logging": {
                "level": "INFO",
                "console": True,
                "file": True,
                "directory": "logs/system",
                "filename": "renix.log",
            },

            "ai": {
                "enabled": True,
                "provider": "auto",
                "model": "auto",
                "temperature": 0.7,
                "max_tokens": 4096,
                "streaming": True,
                "fallback_enabled": True,
                "conversation_context": True,
                "reasoning": True,
                "planning": True,
                "tool_use": True,
                "agent_system": True,
            },

            "voice": {
                "enabled": True,
                "wake_word": "renix",
                "clap_activation": True,
                "clap_count": 2,
                "continuous_conversation": True,
                "interruption": True,
                "barge_in": True,
                "language": "auto",
                "microphone_index": None,
                "speaker_index": None,
                "input_sample_rate": 16000,
                "output_sample_rate": 22050,
                "energy_threshold": 300,
                "silence_timeout": 1.2,
            },

            "speech": {
                "enabled": True,
                "engine": "auto",
                "voice": "default",
                "rate": 175,
                "volume": 1.0,
                "pitch": 1.0,
                "streaming": True,
                "emotion": True,
            },

            "vision": {
                "enabled": True,
                "camera_index": 0,
                "camera_width": 1280,
                "camera_height": 720,
                "camera_fps": 30,

                "face_detection": True,
                "face_recognition": True,
                "face_landmarks": True,

                "object_detection": True,
                "scene_understanding": True,
                "image_understanding": True,

                "hand_tracking": True,
                "hand_landmarks": True,
                "pose_tracking": True,

                "ocr": True,
                "screen_understanding": True,

                "gesture_confidence": 0.75,
                "face_confidence": 0.75,
                "object_confidence": 0.65,
            },

            "gestures": {
                "enabled": True,

                "pinch": True,
                "grab": True,
                "release": True,
                "drag": True,

                "zoom": True,
                "rotate": True,

                "swipe": True,
                "point": True,
                "palm": True,

                "two_hand": True,
                "fist": True,
                "open_hand": True,

                "cursor_control": True,
                "window_control": True,
                "file_control": True,

                "gesture_debounce_ms": 250,
                "gesture_sensitivity": 1.0,
            },

            "memory": {
                "enabled": True,

                "short_term": True,
                "long_term": True,

                "episodic": True,
                "semantic": True,
                "procedural": True,

                "conversation_memory": True,
                "project_memory": True,
                "task_memory": True,

                "vector_search": True,
                "semantic_search": True,

                "maximum_context_items": 100,
                "memory_retention_days": 3650,

                "database": {
                    "type": "sqlite",
                    "path": "data/memory/renix_memory.db",
                },
            },

            "computer": {
                "enabled": True,

                "mouse_control": True,
                "keyboard_control": True,

                "screen_control": True,
                "application_control": True,

                "clipboard_control": True,
                "window_control": True,

                "screenshot": True,
                "screen_recording": True,

                "process_control": True,
                "system_commands": True,

                "confirmation_required": True,
            },

            "files": {
                "enabled": True,

                "search": True,
                "semantic_search": True,

                "copy": True,
                "move": True,
                "rename": True,
                "delete": True,

                "create": True,
                "read": True,
                "write": True,

                "watcher": True,

                "confirmation_required": True,

                "maximum_search_results": 100,
            },

            "browser": {
                "enabled": True,

                "search": True,
                "navigation": True,
                "page_reading": True,

                "page_interaction": True,
                "form_filling": True,

                "downloads": True,
                "tabs": True,

                "automation": True,

                "confirmation_required": True,
            },

            "research": {
                "enabled": True,

                "web_search": True,
                "source_ranking": True,
                "fact_checking": True,
                "cross_validation": True,
                "citations": True,

                "summarization": True,
                "comparison": True,

                "maximum_sources": 10,
            },

            "coding": {
                "enabled": True,

                "code_generation": True,
                "code_analysis": True,
                "debugging": True,

                "testing": True,
                "building": True,

                "file_editing": True,
                "project_understanding": True,

                "terminal": True,

                "git": True,

                "sandbox": True,

                "confirmation_required": True,
            },

            "automation": {
                "enabled": True,

                "scheduled_tasks": True,
                "workflows": True,

                "multi_step_tasks": True,
                "background_tasks": True,

                "proactive_actions": True,

                "task_retry": True,
                "maximum_retries": 3,

                "confirmation_required": True,
            },

            "security": {
                "enabled": True,

                "authentication": True,
                "permissions": True,

                "confirmation_required": True,
                "audit_logging": True,

                "sandbox": True,
                "emergency_lock": True,

                "destructive_action_confirmation": True,
                "external_action_confirmation": True,

                "maximum_command_risk": "medium",
            },

            "ui": {
                "enabled": True,

                "holographic_mode": True,
                "transparent_interface": True,

                "gesture_interface": True,
                "voice_interface": True,

                "animations": True,
                "particles": True,
                "hud": True,

                "three_dimensional": True,

                "fullscreen": True,
                "always_on_top": False,

                "theme": "holographic",
                "fps": 60,

                "cursor": True,
                "notifications": True,
            },

            "monitoring": {
                "enabled": True,

                "cpu": True,
                "gpu": True,
                "ram": True,
                "disk": True,
                "network": True,
                "battery": True,

                "processes": True,
                "temperatures": True,

                "update_interval": 2.0,
            },

            "notifications": {
                "enabled": True,

                "desktop": True,
                "voice": True,

                "priority_filter": "normal",

                "sound": True,
                "duration": 5,
            },

            "education": {
                "enabled": True,

                "study_planning": True,
                "question_solving": True,
                "explanations": True,

                "revision": True,
                "flashcards": True,

                "progress_tracking": True,
            },

            "sports": {
                "enabled": True,

                "cricket": True,
                "fitness": True,

                "performance_tracking": True,
                "statistics": True,
            },

            "media": {
                "enabled": True,

                "music": True,
                "video": True,

                "playback_control": True,
                "volume_control": True,

                "media_search": True,
            },

            "personal": {
                "enabled": True,

                "preferences": True,
                "routines": True,

                "goals": True,
                "reminders": True,

                "personalization": True,
            },

            "analytics": {
                "enabled": True,

                "usage_tracking": True,
                "performance_tracking": True,

                "error_tracking": True,

                "privacy_mode": True,
            },

            "devices": {
                "enabled": True,

                "bluetooth": True,
                "wifi": True,

                "usb": True,
                "mqtt": True,
                "websocket": True,

                "device_discovery": True,
                "device_control": True,
            },

            "robotics": {
                "enabled": True,

                "safety_required": True,
                "emergency_stop": True,

                "simulation_mode": True,
                "physical_control": False,
            },

            "database": {
                "type": "sqlite",

                "main": {
                    "path": "data/database/renix.db",
                },

                "cache": {
                    "path": "data/database/cache.db",
                },
            },

            "network": {
                "enabled": True,

                "timeout": 30,
                "connect_timeout": 10,

                "max_retries": 3,

                "verify_ssl": True,
            },

            "performance": {
                "async_operations": True,

                "worker_threads": 4,
                "worker_processes": 2,

                "cache_enabled": True,

                "maximum_concurrent_tasks": 20,
            },
        }

    # ------------------------------------------------------------------
    # BOOTSTRAP
    # ------------------------------------------------------------------

    def bootstrap(self) -> None:
        """
        Perform a lightweight synchronous configuration load.

        This exists so main.py can obtain logging settings before
        the asynchronous RENIX runtime begins.

        Loading order (lowest → highest priority):
            1. Built-in defaults
            2. config/ component YAML files (ai.yaml, voice.yaml, …)
            3. config.yaml (root-level overrides)
            4. .env / environment variables
        """

        self._config = copy.deepcopy(
            self._defaults
        )

        self._load_environment_file()

        # ── Component YAML files in config/ directory ──────────────────
        component_dir = self.root_dir / "config"
        if component_dir.is_dir():
            for yaml_path in sorted(component_dir.glob("*.yaml")):
                try:
                    component_config = self._read_yaml(yaml_path)
                    self._deep_merge(
                        self._config,
                        component_config,
                    )
                except Exception:
                    self.logger.warning(
                        "Could not load component config: %s",
                        yaml_path.name,
                    )

        # ── Root config.yaml (highest file-level priority) ─────────────
        if self.config_file.exists():
            try:

                yaml_config = self._read_yaml(
                    self.config_file
                )

                self._deep_merge(
                    self._config,
                    yaml_config,
                )

            except Exception:

                self.logger.exception(
                    "Failed to load config.yaml during bootstrap."
                )

        self._apply_environment_overrides()

        self._validate()

    # ------------------------------------------------------------------
    # ASYNC INITIALIZATION
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Fully initialize configuration.

        This method is asynchronous so it fits naturally into the
        RENIX application lifecycle.
        """

        self.bootstrap()

        self._initialized = True

        self.logger.info(
            "Configuration initialized."
        )

    # ------------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """
        Shut down configuration manager.

        Configuration is intentionally kept in memory until process
        termination, so this method mainly resets lifecycle state.
        """

        self._initialized = False

    # ------------------------------------------------------------------
    # ENVIRONMENT FILE
    # ------------------------------------------------------------------

    def _load_environment_file(
        self,
    ) -> None:
        """
        Load values from .env if it exists.
        """

        if not self.env_file.exists():
            return

        load_dotenv(
            dotenv_path=self.env_file,
            override=False,
        )

    # ------------------------------------------------------------------
    # YAML
    # ------------------------------------------------------------------

    @staticmethod
    def _read_yaml(
        path: Path,
    ) -> dict[str, Any]:
        """
        Read a YAML configuration file.

        Automatically strips markdown code fences (```yaml ... ```)
        that may wrap the file content, so the config/ files authored
        with markdown formatting are parsed cleanly.
        """

        import re as _re

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw = file.read()

        # Strip leading ```yaml or ``` fence
        raw = _re.sub(r"^```(?:yaml)?\s*\r?\n", "", raw, count=1)
        # Strip trailing ``` fence
        raw = _re.sub(r"\r?\n```\s*$", "", raw)

        result = yaml.safe_load(raw)

        if result is None:
            return {}

        if not isinstance(
            result,
            dict,
        ):

            raise ValueError(
                "The root of config.yaml must be a mapping."
            )

        return result

    # ------------------------------------------------------------------
    # DEEP MERGE
    # ------------------------------------------------------------------

    @classmethod
    def _deep_merge(
        cls,
        target: dict[str, Any],
        source: dict[str, Any],
    ) -> None:
        """
        Recursively merge source into target.

        Nested dictionaries are merged instead of completely replaced.
        """

        for key, value in source.items():

            if (
                key in target
                and isinstance(
                    target[key],
                    dict,
                )
                and isinstance(
                    value,
                    dict,
                )
            ):

                cls._deep_merge(
                    target[key],
                    value,
                )

            else:

                target[key] = copy.deepcopy(
                    value
                )

    # ------------------------------------------------------------------
    # ENVIRONMENT OVERRIDES
    # ------------------------------------------------------------------

    def _apply_environment_overrides(
        self,
    ) -> None:
        """
        Apply explicit environment-variable overrides.

        Example:

            RENIX_NAME=Friday

        becomes:

            renix.name = Friday
        """

        mappings = {
            "RENIX_NAME": "renix.name",
            "RENIX_VERSION": "renix.version",
            "RENIX_ENV": "renix.environment",
            "RENIX_DEBUG": "renix.debug",

            "LOG_LEVEL": "logging.level",

            "AI_PROVIDER": "ai.provider",
            "AI_MODEL": "ai.model",
            "AI_API_KEY": "ai.api_key",

            "RENIX_WAKE_WORD": "voice.wake_word",

            "VOICE_ENABLED": "voice.enabled",
            "VISION_ENABLED": "vision.enabled",
            "GESTURE_ENABLED": "gestures.enabled",

            "MEMORY_ENABLED": "memory.enabled",

            "WEB_ENABLED": "browser.enabled",

            "COMPUTER_CONTROL_ENABLED":
                "computer.enabled",

            "FILE_CONTROL_ENABLED":
                "files.enabled",

            "AUTOMATION_ENABLED":
                "automation.enabled",

            "HOLOGRAPHIC_UI_ENABLED":
                "ui.holographic_mode",
        }

        for environment_name, config_path in mappings.items():

            raw_value = os.getenv(
                environment_name
            )

            if raw_value is None:
                continue

            converted_value = (
                self._convert_environment_value(
                    raw_value
                )
            )

            self.set(
                config_path,
                converted_value,
                create_missing=True,
                publish=False,
            )

    # ------------------------------------------------------------------
    # VALUE CONVERSION
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_environment_value(
        value: str,
    ) -> Any:
        """
        Convert environment strings into useful Python values.
        """

        normalized = value.strip().lower()

        if normalized in {
            "true",
            "yes",
            "on",
            "1",
        }:

            return True

        if normalized in {
            "false",
            "no",
            "off",
            "0",
        }:

            return False

        if normalized in {
            "none",
            "null",
        }:

            return None

        # Integer
        try:
            if (
                "." not in value
                and "e" not in normalized
            ):

                return int(value)

        except ValueError:
            pass

        # Float
        try:
            return float(value)

        except ValueError:
            pass

        return value

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------

    def get(
        self,
        path: str,
        default: Any = None,
    ) -> Any:
        """
        Get a nested configuration value.

        Example:

            config.get("ai.model")

            config.get(
                "voice.wake_word",
                "renix",
            )
        """

        if not path:
            return default

        current: Any = self._config

        for part in path.split("."):

            if not isinstance(
                current,
                dict,
            ):

                return default

            if part not in current:
                return default

            current = current[part]

        return copy.deepcopy(
            current
        )

    # ------------------------------------------------------------------
    # SET
    # ------------------------------------------------------------------

    def set(
        self,
        path: str,
        value: Any,
        *,
        create_missing: bool = True,
        publish: bool = True,
    ) -> None:
        """
        Set a nested configuration value.

        Example:

            config.set(
                "ui.theme",
                "holographic",
            )
        """

        if not path:
            raise ValueError(
                "Configuration path cannot be empty."
            )

        parts = [
            part.strip()
            for part in path.split(".")
            if part.strip()
        ]

        if not parts:
            raise ValueError(
                "Configuration path is invalid."
            )

        current = self._config

        for part in parts[:-1]:

            if part not in current:

                if not create_missing:

                    raise KeyError(
                        f"Configuration section does not exist: {part}"
                    )

                current[part] = {}

            if not isinstance(
                current[part],
                dict,
            ):

                if not create_missing:

                    raise TypeError(
                        f"Configuration path '{part}' is not a mapping."
                    )

                current[part] = {}

            current = current[part]

        current[parts[-1]] = copy.deepcopy(
            value
        )

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------

    def delete(
        self,
        path: str,
    ) -> bool:
        """
        Delete a configuration value.

        Returns True if something was deleted.
        """

        if not path:
            return False

        parts = [
            part.strip()
            for part in path.split(".")
            if part.strip()
        ]

        if not parts:
            return False

        current: Any = self._config

        for part in parts[:-1]:

            if not isinstance(
                current,
                dict,
            ):

                return False

            if part not in current:

                return False

            current = current[part]

        if not isinstance(
            current,
            dict,
        ):

            return False

        final_key = parts[-1]

        if final_key not in current:
            return False

        del current[final_key]

        return True

    # ------------------------------------------------------------------
    # SECTION
    # ------------------------------------------------------------------

    def get_section(
        self,
        section: str,
    ) -> dict[str, Any]:
        """
        Return an entire configuration section.
        """

        value = self.get(
            section,
            {},
        )

        if not isinstance(
            value,
            dict,
        ):

            return {}

        return value

    # ------------------------------------------------------------------
    # COMPLETE CONFIGURATION
    # ------------------------------------------------------------------

    def all(
        self,
    ) -> dict[str, Any]:
        """
        Return a deep copy of the entire configuration.
        """

        return copy.deepcopy(
            self._config
        )

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    def save(
        self,
        path: Path | None = None,
    ) -> None:
        """
        Save the current configuration to YAML.

        By default this overwrites config.yaml.
        """

        destination = (
            Path(path)
            if path is not None
            else self.config_file
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with destination.open(
            "w",
            encoding="utf-8",
        ) as file:

            yaml.safe_dump(
                self._config,
                file,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(
        self,
    ) -> None:
        """
        Reset in-memory configuration to defaults.
        """

        self._config = copy.deepcopy(
            self._defaults
        )

        self._apply_environment_overrides()

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def _validate(
        self,
    ) -> None:
        """
        Perform basic configuration validation.

        This intentionally avoids rejecting unknown settings so that
        future RENIX modules can introduce their own configuration
        sections without breaking older versions.
        """

        level = str(
            self.get(
                "logging.level",
                "INFO",
            )
        ).upper()

        valid_levels = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        if level not in valid_levels:

            self.logger.warning(
                "Unknown logging level '%s'. "
                "Falling back to INFO.",
                level,
            )

            self.set(
                "logging.level",
                "INFO",
                publish=False,
            )

        temperature = self.get(
            "ai.temperature",
            0.7,
        )

        try:

            temperature = float(
                temperature
            )

        except (
            TypeError,
            ValueError,
        ):

            temperature = 0.7

        temperature = max(
            0.0,
            min(
                2.0,
                temperature,
            ),
        )

        self.set(
            "ai.temperature",
            temperature,
            publish=False,
        )

        confidence = self.get(
            "vision.gesture_confidence",
            0.75,
        )

        try:

            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError,
        ):

            confidence = 0.75

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        self.set(
            "vision.gesture_confidence",
            confidence,
            publish=False,
        )

    # ------------------------------------------------------------------
    # CONVENIENCE PROPERTIES
    # ------------------------------------------------------------------

    @property
    def initialized(
        self,
    ) -> bool:

        return self._initialized

    @property
    def name(
        self,
    ) -> str:

        return str(
            self.get(
                "renix.name",
                "RENIX",
            )
        )

    @property
    def version(
        self,
    ) -> str:

        return str(
            self.get(
                "renix.version",
                "1.0.0",
            )
        )

    @property
    def environment(
        self,
    ) -> str:

        return str(
            self.get(
                "renix.environment",
                "development",
            )
        )

    @property
    def debug(
        self,
    ) -> bool:

        return bool(
            self.get(
                "renix.debug",
                False,
            )
        )

    # ------------------------------------------------------------------
    # FEATURE CHECK
    # ------------------------------------------------------------------

    def is_enabled(
        self,
        feature: str,
        default: bool = False,
    ) -> bool:
        """
        Check whether a RENIX feature is enabled.

        Example:

            config.is_enabled("voice.enabled")
        """

        value = self.get(
            feature,
            default,
        )

        if isinstance(
            value,
            bool,
        ):

            return value

        if isinstance(
            value,
            str,
        ):

            return (
                value.strip().lower()
                in {
                    "true",
                    "yes",
                    "on",
                    "1",
                }
            )

        return bool(value)

    # ------------------------------------------------------------------
    # SECRET ACCESS
    # ------------------------------------------------------------------

    def get_secret(
        self,
        name: str,
        default: str | None = None,
    ) -> str | None:
        """
        Retrieve a secret from the environment.

        Secrets are deliberately not stored in config.yaml.

        Example:

            config.get_secret("AI_API_KEY")
        """

        value = os.getenv(
            name
        )

        if value is None:
            return default

        value = value.strip()

        if not value:
            return default

        return value

    # ------------------------------------------------------------------
    # PATH HELPERS
    # ------------------------------------------------------------------

    def resolve_path(
        self,
        configured_path: str,
    ) -> Path:
        """
        Resolve a path relative to the RENIX root directory.

        Absolute paths remain absolute.
        """

        path = Path(
            configured_path
        )

        if path.is_absolute():
            return path

        return (
            self.root_dir / path
        ).resolve()

    # ------------------------------------------------------------------
    # REPRESENTATION
    # ------------------------------------------------------------------

    def __repr__(
        self,
    ) -> str:

        return (
            f"ConfigManager("
            f"name={self.name!r}, "
            f"version={self.version!r}, "
            f"environment={self.environment!r}, "
            f"initialized={self.initialized!r}"
            f")"
        )


