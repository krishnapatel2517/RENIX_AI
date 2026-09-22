"""
RENIX Alexa Smart Home Provider
===============================

Adapter for Amazon Alexa smart-home integrations.

The underlying Alexa API/client is injected so RENIX does not depend
on a particular SDK implementation.
"""

from __future__ import annotations

from typing import Any

from .smart_home_provider import SmartHomeProvider


class AlexaProvider(SmartHomeProvider):
    """Amazon Alexa smart-home provider."""

    name = "alexa"

    def __init__(
        self,
        *,
        client: Any = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            config=config
        )

        self.client = client

    # ========================================================
    # INTERNAL
    # ========================================================

    def _require_client(self) -> Any:
        if self.client is None:
            raise RuntimeError(
                "Alexa client is not configured."
            )

        return self.client

    def _call(
        self,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        client = self._require_client()

        function = getattr(
            client,
            method,
            None,
        )

        if not callable(function):
            raise RuntimeError(
                f"Alexa client does not implement "
                f"'{method}'."
            )

        return function(
            *args,
            **kwargs,
        )

    # ========================================================
    # CONNECTION
    # ========================================================

    def connect(self) -> bool:
        """Connect to the Alexa integration."""

        client = self._require_client()

        connect = getattr(
            client,
            "connect",
            None,
        )

        if callable(connect):
            result = connect()

            self._connected = (
                True
                if result is None
                else bool(result)
            )
        else:
            self._connected = True

        return self._connected

    def disconnect(self) -> None:
        """Disconnect from Alexa."""

        if self.client is not None:
            disconnect = getattr(
                self.client,
                "disconnect",
                None,
            )

            if callable(disconnect):
                disconnect()

        self._connected = False

    # ========================================================
    # DEVICES
    # ========================================================

    def list_devices(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return Alexa-discovered smart-home devices."""

        self.ensure_connected()

        result = self._call(
            "list_devices",
            **kwargs,
        )

        if isinstance(
            result,
            dict,
        ):
            result = result.get(
                "devices",
                [],
            )

        return list(
            result or []
        )

    def get_device(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return information about an Alexa device."""

        self.ensure_connected()

        if not device_id.strip():
            raise ValueError(
                "device_id cannot be empty."
            )

        result = self._call(
            "get_device",
            device_id,
            **kwargs,
        )

        if isinstance(
            result,
            dict,
        ):
            return result

        return {
            "id": device_id,
            "state": result,
        }

    def get_state(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return device state."""

        self.ensure_connected()

        result = self._call(
            "get_state",
            device_id,
            **kwargs,
        )

        if isinstance(
            result,
            dict,
        ):
            return result

        return {
            "device_id": device_id,
            "state": result,
        }

    # ========================================================
    # GENERIC COMMANDS
    # ========================================================

    def call_service(
        self,
        service: str,
        *,
        device_id: str | None = None,
        entity_id: str | None = None,
        data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute an Alexa smart-home command."""

        self.ensure_connected()

        target = (
            device_id
            or entity_id
        )

        if not target:
            raise ValueError(
                "device_id or entity_id is required."
            )

        payload = dict(
            data or {}
        )

        payload.update(
            kwargs
        )

        return self._call(
            "execute_command",
            service,
            target,
            payload,
        )

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
            data=kwargs,
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
            data=kwargs,
        )

    def set_light_brightness(
        self,
        device_id: str,
        brightness: float,
        **kwargs: Any,
    ) -> Any:
        """Set light brightness."""

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
                **kwargs,
            },
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
                **kwargs,
            },
        )

    # ========================================================
    # SWITCHES / PLUGS
    # ========================================================

    def turn_switch_on(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Turn a switch or plug on."""

        return self.call_service(
            "switch.turn_on",
            device_id=device_id,
            data=kwargs,
        )

    def turn_switch_off(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Turn a switch or plug off."""

        return self.call_service(
            "switch.turn_off",
            device_id=device_id,
            data=kwargs,
        )

    def toggle_switch(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
        """Toggle a switch or plug."""

        return self.call_service(
            "switch.toggle",
            device_id=device_id,
            data=kwargs,
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
            data=kwargs,
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
            data=kwargs,
        )

    def set_fan_speed(
        self,
        device_id: str,
        speed: float,
        **kwargs: Any,
    ) -> Any:
        """Set fan speed."""

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
                **kwargs,
            },
        )

    # ========================================================
    # CLIMATE / AC
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
                **kwargs,
            },
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
                "mode cannot be empty."
            )

        return self.call_service(
            "climate.set_mode",
            device_id=device_id,
            data={
                "mode": mode,
                **kwargs,
            },
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
            data=kwargs,
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
            data=kwargs,
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
            data=kwargs,
        )

    def set_media_volume(
        self,
        device_id: str,
        volume: float,
        **kwargs: Any,
    ) -> Any:
        """Set media volume."""

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
                **kwargs,
            },
        )

    # ========================================================
    # ROUTINES / SCENES
    # ========================================================

    def list_routines(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return Alexa routines."""

        self.ensure_connected()

        result = self._call(
            "list_routines",
            **kwargs,
        )

        if isinstance(
            result,
            dict,
        ):
            result = result.get(
                "routines",
                [],
            )

        return list(
            result or []
        )

    def trigger_routine(
        self,
        routine_id: str,
        **kwargs: Any,
    ) -> Any:
        """Trigger an Alexa routine."""

        if not routine_id.strip():
            raise ValueError(
                "routine_id cannot be empty."
            )

        return self.call_service(
            "routine.trigger",
            entity_id=routine_id,
            data=kwargs,
        )

    def list_scenes(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return Alexa scenes."""

        self.ensure_connected()

        result = self._call(
            "list_scenes",
            **kwargs,
        )

        if isinstance(
            result,
            dict,
        ):
            result = result.get(
                "scenes",
                [],
            )

        return list(
            result or []
        )

    def activate_scene(
        self,
        scene_id: str,
        **kwargs: Any,
    ) -> Any:
        """Activate an Alexa scene."""

        return self.call_service(
            "scene.activate",
            entity_id=scene_id,
            data=kwargs,
        )

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(self) -> bool:
        """Check Alexa provider health."""

        try:
            if self.client is None:
                return False

            health = getattr(
                self.client,
                "health_check",
                None,
            )

            if callable(health):
                return bool(
                    health()
                )

            return self.is_connected()

        except Exception:
            return False

    # ========================================================
    # CAPABILITIES
    # ========================================================

    def capabilities(
        self,
    ) -> dict[str, bool]:
        return {
            "devices": True,
            "state": True,
            "service_calls": True,
            "lights": True,
            "switches": True,
            "plugs": True,
            "fans": True,
            "climate": True,
            "media": True,
            "scenes": True,
            "routines": True,
        }

    # ========================================================
    # HELPERS
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
            value = int(
                value
            )

            if not 0 <= value <= 255:
                raise ValueError(
                    f"{name} must be between 0 and 255."
                )


__all__ = [
    "AlexaProvider",
]


