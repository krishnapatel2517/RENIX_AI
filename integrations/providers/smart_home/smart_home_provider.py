"""
RENIX Smart Home Provider
=========================

Base interface for smart-home integrations.

Concrete providers such as Home Assistant, MQTT-based systems,
or other platforms should implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SmartHomeProvider(ABC):
    """
    Abstract base class for RENIX smart-home providers.

    A provider is responsible for communicating with an external
    smart-home platform. RENIX can then use the same interface for:

        - lights
        - fans
        - air conditioners
        - televisions
        - speakers
        - plugs
        - sensors
        - cameras
        - scenes
        - automations
    """

    name: str = "smart_home"

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or {}

        self._connected = False

    # ========================================================
    # CONNECTION
    # ========================================================

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the smart-home platform."""

        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the smart-home platform."""

        raise NotImplementedError

    def is_connected(self) -> bool:
        """Return connection state."""

        return self._connected

    def ensure_connected(self) -> None:
        """Connect when the provider is not connected."""

        if not self.is_connected():
            if not self.connect():
                raise ConnectionError(
                    f"Unable to connect to "
                    f"{self.name} smart-home provider."
                )

    # ========================================================
    # DEVICES
    # ========================================================

    @abstractmethod
    def list_devices(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return available smart-home devices."""

        raise NotImplementedError

    @abstractmethod
    def get_device(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return information about a device."""

        raise NotImplementedError

    @abstractmethod
    def get_state(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return the current state of a device."""

        raise NotImplementedError

    # ========================================================
    # COMMANDS
    # ========================================================

    @abstractmethod
    def call_service(
        self,
        service: str,
        *,
        device_id: str | None = None,
        entity_id: str | None = None,
        data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a smart-home service/action.

        Example services:

            light.turn_on
            light.turn_off
            fan.turn_on
            switch.turn_off
            climate.set_temperature
        """

        raise NotImplementedError

    # ========================================================
    # LIGHTS
    # ========================================================

    def turn_light_on(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Turn a light on."""

        return self.call_service(
            "light.turn_on",
            device_id=device_id,
            **kwargs,
        )

    def turn_light_off(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Turn a light off."""

        return self.call_service(
            "light.turn_off",
            device_id=device_id,
            **kwargs,
        )

    def set_light_brightness(
        self,
        device_id: str,
        brightness: float,
        **kwargs: Any,
    ) -> Any:
        """Set light brightness from 0.0 to 1.0."""

        brightness = float(
            brightness
        )

        if brightness > 1.0:
            brightness /= 100.0

        brightness = max(
            0.0,
            min(
                1.0,
                brightness,
            ),
        )

        return self.call_service(
            "light.set_brightness",
            device_id=device_id,
            data={
                "brightness": brightness,
            },
            **kwargs,
        )

    def set_light_color(
        self,
        device_id: str,
        red: int,
        green: int,
        blue: int,
        **kwargs: Any,
    ) -> Any:
        """Set RGB light color."""

        self._validate_rgb(
            red,
            green,
            blue,
        )

        return self.call_service(
            "light.set_color",
            device_id=device_id,
            data={
                "red": red,
                "green": green,
                "blue": blue,
            },
            **kwargs,
        )

    # ========================================================
    # SWITCHES / PLUGS
    # ========================================================

    def turn_switch_on(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Turn a switch or smart plug on."""

        return self.call_service(
            "switch.turn_on",
            device_id=device_id,
            **kwargs,
        )

    def turn_switch_off(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Turn a switch or smart plug off."""

        return self.call_service(
            "switch.turn_off",
            device_id=device_id,
            **kwargs,
        )

    def toggle_switch(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Toggle a switch or smart plug."""

        return self.call_service(
            "switch.toggle",
            device_id=device_id,
            **kwargs,
        )

    # ========================================================
    # FANS
    # ========================================================

    def turn_fan_on(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Turn a fan on."""

        return self.call_service(
            "fan.turn_on",
            device_id=device_id,
            **kwargs,
        )

    def turn_fan_off(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Turn a fan off."""

        return self.call_service(
            "fan.turn_off",
            device_id=device_id,
            **kwargs,
        )

    def set_fan_speed(
        self,
        device_id: str,
        speed: float,
        **kwargs: Any,
    ) -> Any:
        """Set fan speed from 0.0 to 1.0."""

        speed = float(
            speed
        )

        if speed > 1.0:
            speed /= 100.0

        speed = max(
            0.0,
            min(
                1.0,
                speed,
            ),
        )

        return self.call_service(
            "fan.set_speed",
            device_id=device_id,
            data={
                "speed": speed,
            },
            **kwargs,
        )

    # ========================================================
    # AIR CONDITIONING
    # ========================================================

    def set_temperature(
        self,
        device_id: str,
        temperature: float,
        **kwargs: Any,
    ) -> Any:
        """Set target temperature."""

        return self.call_service(
            "climate.set_temperature",
            device_id=device_id,
            data={
                "temperature": float(
                    temperature
                ),
            },
            **kwargs,
        )

    def set_hvac_mode(
        self,
        device_id: str,
        mode: str,
        **kwargs: Any,
    ) -> Any:
        """Set HVAC mode."""

        mode = mode.strip().lower()

        if not mode:
            raise ValueError(
                "HVAC mode cannot be empty."
            )

        return self.call_service(
            "climate.set_mode",
            device_id=device_id,
            data={
                "mode": mode,
            },
            **kwargs,
        )

    # ========================================================
    # MEDIA / TV
    # ========================================================

    def media_play(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Start media playback."""

        return self.call_service(
            "media.play",
            device_id=device_id,
            **kwargs,
        )

    def media_pause(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Pause media playback."""

        return self.call_service(
            "media.pause",
            device_id=device_id,
            **kwargs,
        )

    def media_stop(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Stop media playback."""

        return self.call_service(
            "media.stop",
            device_id=device_id,
            **kwargs,
        )

    def set_media_volume(
        self,
        device_id: str,
        volume: float,
        **kwargs: Any,
    ) -> Any:
        """Set media volume from 0.0 to 1.0."""

        volume = float(
            volume
        )

        if volume > 1.0:
            volume /= 100.0

        volume = max(
            0.0,
            min(
                1.0,
                volume,
            ),
        )

        return self.call_service(
            "media.set_volume",
            device_id=device_id,
            data={
                "volume": volume,
            },
            **kwargs,
        )

    # ========================================================
    # SCENES
    # ========================================================

    def list_scenes(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return available scenes."""

        try:
            result = self.call_service(
                "scene.list",
                **kwargs,
            )
        except (
            NotImplementedError,
            RuntimeError,
        ):
            return []

        if isinstance(
            result,
            list,
        ):
            return result

        return []

    def activate_scene(
        self,
        scene_id: str,
        **kwargs: Any,
    ) -> Any:
        """Activate a smart-home scene."""

        if not scene_id.strip():
            raise ValueError(
                "scene_id cannot be empty."
            )

        return self.call_service(
            "scene.activate",
            entity_id=scene_id,
            **kwargs,
        )

    # ========================================================
    # AUTOMATIONS
    # ========================================================

    def list_automations(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return available automations."""

        try:
            result = self.call_service(
                "automation.list",
                **kwargs,
            )
        except (
            NotImplementedError,
            RuntimeError,
        ):
            return []

        if isinstance(
            result,
            list,
        ):
            return result

        return []

    def trigger_automation(
        self,
        automation_id: str,
        **kwargs: Any,
    ) -> Any:
        """Trigger an automation."""

        if not automation_id.strip():
            raise ValueError(
                "automation_id cannot be empty."
            )

        return self.call_service(
            "automation.trigger",
            entity_id=automation_id,
            **kwargs,
        )

    # ========================================================
    # GENERIC DEVICE CONTROL
    # ========================================================

    def turn_on(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Generic device turn-on operation."""

        return self.call_service(
            "device.turn_on",
            device_id=device_id,
            **kwargs,
        )

    def turn_off(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Generic device turn-off operation."""

        return self.call_service(
            "device.turn_off",
            device_id=device_id,
            **kwargs,
        )

    def toggle(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Generic device toggle operation."""

        return self.call_service(
            "device.toggle",
            device_id=device_id,
            **kwargs,
        )

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(self) -> bool:
        """Check provider health."""

        try:
            return self.is_connected()
        except Exception:
            return False

    def capabilities(self) -> dict[str, bool]:
        """Return supported capabilities."""

        return {
            "devices": True,
            "state": True,
            "lights": True,
            "switches": True,
            "plugs": True,
            "fans": True,
            "climate": True,
            "media": True,
            "scenes": True,
            "automations": True,
        }

    # ========================================================
    # UTILITIES
    # ========================================================

    @staticmethod
    def _validate_rgb(
        red: int,
        green: int,
        blue: int,
    ) -> None:
        for name, value in (
            ("red", red),
            ("green", green),
            ("blue", blue),
        ):
            value = int(value)

            if not 0 <= value <= 255:
                raise ValueError(
                    f"{name} must be between 0 and 255."
                )


__all__ = [
    "SmartHomeProvider",
]


