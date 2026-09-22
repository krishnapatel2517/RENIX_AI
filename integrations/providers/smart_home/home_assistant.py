"""
RENIX Home Assistant Provider
============================

Home Assistant integration for RENIX.

The HTTP/WebSocket transport is injected through `client`, keeping
this provider independent of a particular Home Assistant library.
"""

from __future__ import annotations

from typing import Any

from .smart_home_provider import SmartHomeProvider


class HomeAssistantProvider(SmartHomeProvider):
    """Provider for Home Assistant."""

    name = "home_assistant"

    def __init__(
        self,
        *,
        client: Any = None,
        base_url: str | None = None,
        token: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            config=config
        )

        self.client = client
        self.base_url = (
            base_url
            or self.config.get(
                "base_url"
            )
        )
        self.token = (
            token
            or self.config.get(
                "token"
            )
        )

    # ========================================================
    # INTERNAL
    # ========================================================

    def _require_client(self) -> Any:
        if self.client is None:
            raise RuntimeError(
                "Home Assistant client is not configured."
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
                f"Home Assistant client does not implement "
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
        """Connect to Home Assistant."""

        if self.client is None:
            raise RuntimeError(
                "Home Assistant client is required."
            )

        connect = getattr(
            self.client,
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
        """Disconnect from Home Assistant."""

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
    # DEVICES / ENTITIES
    # ========================================================

    def list_devices(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return Home Assistant entities."""

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
                result.get(
                    "entities",
                    [],
                ),
            )

        return list(
            result or []
        )

    def get_device(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a Home Assistant entity."""

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

        if not isinstance(
            result,
            dict,
        ):
            return {
                "id": device_id,
                "state": result,
            }

        return result

    def get_state(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return entity state."""

        self.ensure_connected()

        if not device_id.strip():
            raise ValueError(
                "device_id cannot be empty."
            )

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
            "entity_id": device_id,
            "state": result,
        }

    # ========================================================
    # SERVICE CALLS
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
        """Call a Home Assistant service."""

        self.ensure_connected()

        service = service.strip()

        if not service:
            raise ValueError(
                "service cannot be empty."
            )

        payload = dict(
            data or {}
        )

        if device_id:
            payload.setdefault(
                "device_id",
                device_id,
            )

        if entity_id:
            payload.setdefault(
                "entity_id",
                entity_id,
            )

        domain, service_name = (
            self._split_service(
                service
            )
        )

        return self._call(
            "call_service",
            domain,
            service_name,
            data=payload,
            **kwargs,
        )

    # ========================================================
    # HOME ASSISTANT SPECIFIC
    # ========================================================

    def get_entities(
        self,
        *,
        domain: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Get entities, optionally filtered by domain."""

        self.ensure_connected()

        result = self._call(
            "get_entities",
            **kwargs,
        )

        if isinstance(
            result,
            dict,
        ):
            result = result.get(
                "entities",
                [],
            )

        entities = list(
            result or []
        )

        if not domain:
            return entities

        domain = domain.strip().lower()

        return [
            entity
            for entity in entities
            if self._entity_domain(
                entity
            ) == domain
        ]

    def get_entity(
        self,
        entity_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get a single Home Assistant entity."""

        self.ensure_connected()

        return self.get_state(
            entity_id,
            **kwargs,
        )

    # ========================================================
    # LIGHTS
    # ========================================================

    def turn_light_on(
        self,
        device_id: str,
        *,
        brightness: float | None = None,
        rgb_color: tuple[int, int, int] | None = None,
        color_temperature: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Turn on a Home Assistant light."""

        data: dict[str, Any] = {}

        if brightness is not None:
            brightness = float(
                brightness
            )

            if brightness <= 1.0:
                brightness = round(
                    brightness * 255
                )

            data[
                "brightness"
            ] = max(
                0,
                min(
                    255,
                    int(brightness),
                ),
            )

        if rgb_color is not None:
            self._validate_rgb(
                *rgb_color
            )

            data[
                "rgb_color"
            ] = list(
                rgb_color
            )

        if color_temperature is not None:
            data[
                "color_temp_kelvin"
            ] = int(
                color_temperature
            )

        return self.call_service(
            "light.turn_on",
            device_id=device_id,
            data=data,
            **kwargs,
        )

    def set_light_brightness(
        self,
        device_id: str,
        brightness: float,
        **kwargs: Any,
    ) -> Any:
        """Set Home Assistant light brightness."""

        brightness = float(
            brightness
        )

        if brightness <= 1.0:
            brightness *= 255

        brightness = max(
            0,
            min(
                255,
                int(brightness),
            ),
        )

        return self.call_service(
            "light.turn_on",
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
        """Set Home Assistant RGB light color."""

        self._validate_rgb(
            red,
            green,
            blue,
        )

        return self.call_service(
            "light.turn_on",
            device_id=device_id,
            data={
                "rgb_color": [
                    red,
                    green,
                    blue,
                ]
            },
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
        """Turn on a fan."""

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
        """Turn off a fan."""

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
        """Set fan percentage."""

        speed = float(
            speed
        )

        if speed <= 1.0:
            speed *= 100

        speed = max(
            0,
            min(
                100,
                int(speed),
            ),
        )

        return self.call_service(
            "fan.set_percentage",
            device_id=device_id,
            data={
                "percentage": speed,
            },
            **kwargs,
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
        """Set Home Assistant climate temperature."""

        return self.call_service(
            "climate.set_temperature",
            device_id=device_id,
            data={
                "temperature": float(
                    temperature
                )
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

        return self.call_service(
            "climate.set_hvac_mode",
            device_id=device_id,
            data={
                "hvac_mode": mode,
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
        """Turn on a switch."""

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
        """Turn off a switch."""

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
        """Toggle a switch."""

        return self.call_service(
            "switch.toggle",
            device_id=device_id,
            **kwargs,
        )

    # ========================================================
    # SCENES
    # ========================================================

    def list_scenes(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """List Home Assistant scenes."""

        return self.get_entities(
            domain="scene",
            **kwargs,
        )

    def activate_scene(
        self,
        scene_id: str,
        **kwargs: Any,
    ) -> Any:
        """Activate a Home Assistant scene."""

        return self.call_service(
            "scene.turn_on",
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
        """List Home Assistant automations."""

        return self.get_entities(
            domain="automation",
            **kwargs,
        )

    def trigger_automation(
        self,
        automation_id: str,
        **kwargs: Any,
    ) -> Any:
        """Trigger a Home Assistant automation."""

        return self.call_service(
            "automation.trigger",
            entity_id=automation_id,
            **kwargs,
        )

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(self) -> bool:
        """Check Home Assistant connection."""

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
        """Return Home Assistant capabilities."""

        return {
            "devices": True,
            "entities": True,
            "state": True,
            "service_calls": True,
            "lights": True,
            "switches": True,
            "plugs": True,
            "fans": True,
            "climate": True,
            "scenes": True,
            "automations": True,
        }

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _split_service(
        service: str,
    ) -> tuple[str, str]:
        if "." not in service:
            raise ValueError(
                "Home Assistant service must use "
                "'domain.service' format."
            )

        domain, service_name = service.split(
            ".",
            1,
        )

        if not domain or not service_name:
            raise ValueError(
                "Invalid Home Assistant service."
            )

        return (
            domain.strip(),
            service_name.strip(),
        )

    @staticmethod
    def _entity_domain(
        entity: dict[str, Any],
    ) -> str | None:
        entity_id = entity.get(
            "entity_id"
        )

        if not entity_id:
            return None

        entity_id = str(
            entity_id
        )

        if "." not in entity_id:
            return None

        return entity_id.split(
            ".",
            1,
        )[0].lower()

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
    "HomeAssistantProvider",
]


