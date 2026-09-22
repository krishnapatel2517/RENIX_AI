"""
RENIX MQTT Smart Home Provider
==============================

Generic MQTT-based smart-home provider.

This provider does not assume a particular MQTT library. An MQTT
client/adapter is injected and is expected to expose connect,
disconnect, publish, subscribe, and optionally request methods.

Example topic structure:

    renix/home/<device_id>/command
    renix/home/<device_id>/state
    renix/home/<device_id>/availability
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable

from .smart_home_provider import SmartHomeProvider


class MQTTProvider(SmartHomeProvider):
    """Generic MQTT smart-home provider."""

    name = "mqtt"

    def __init__(
        self,
        *,
        client: Any = None,
        broker: str | None = None,
        port: int = 1883,
        username: str | None = None,
        password: str | None = None,
        base_topic: str = "renix/home",
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            config=config
        )

        self.client = client

        self.broker = (
            broker
            or self.config.get(
                "broker"
            )
        )

        self.port = int(
            self.config.get(
                "port",
                port,
            )
        )

        self.username = (
            username
            or self.config.get(
                "username"
            )
        )

        self.password = (
            password
            or self.config.get(
                "password"
            )
        )

        self.base_topic = (
            base_topic
            or self.config.get(
                "base_topic",
                "renix/home",
            )
        ).strip(
            "/"
        )

        self._states: dict[
            str,
            dict[str, Any],
        ] = {}

        self._availability: dict[
            str,
            bool,
        ] = {}

        self._state_callbacks: list[
            Callable[
                [str, dict[str, Any]],
                None,
            ]
        ] = []

        self._lock = threading.RLock()

    # ========================================================
    # INTERNAL
    # ========================================================

    def _require_client(self) -> Any:
        if self.client is None:
            raise RuntimeError(
                "MQTT client is not configured."
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
                f"MQTT client does not implement "
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
        """Connect to the MQTT broker."""

        client = self._require_client()

        if self.username:
            username_setter = getattr(
                client,
                "username_pw_set",
                None,
            )

            if callable(username_setter):
                username_setter(
                    self.username,
                    self.password,
                )

        connect = getattr(
            client,
            "connect",
            None,
        )

        if not callable(connect):
            raise RuntimeError(
                "MQTT client does not implement 'connect'."
            )

        result = connect(
            self.broker,
            self.port,
        )

        self._connected = (
            True
            if result is None
            else bool(result)
        )

        self._subscribe_to_base_topics()

        return self._connected

    def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""

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
    # TOPICS
    # ========================================================

    def device_topic(
        self,
        device_id: str,
        suffix: str = "command",
    ) -> str:
        """Build a device MQTT topic."""

        device_id = str(
            device_id
        ).strip(
            "/"
        )

        suffix = str(
            suffix
        ).strip(
            "/"
        )

        return (
            f"{self.base_topic}/"
            f"{device_id}/"
            f"{suffix}"
        )

    def command_topic(
        self,
        device_id: str,
    ) -> str:
        return self.device_topic(
            device_id,
            "command",
        )

    def state_topic(
        self,
        device_id: str,
    ) -> str:
        return self.device_topic(
            device_id,
            "state",
        )

    def availability_topic(
        self,
        device_id: str,
    ) -> str:
        return self.device_topic(
            device_id,
            "availability",
        )

    # ========================================================
    # PUBLISH
    # ========================================================

    def publish(
        self,
        topic: str,
        payload: Any,
        *,
        retain: bool = False,
        qos: int = 0,
    ) -> Any:
        """Publish a payload to MQTT."""

        self.ensure_connected()

        if isinstance(
            payload,
            (dict, list, tuple),
        ):
            payload = json.dumps(
                payload
            )

        elif payload is None:
            payload = ""

        else:
            payload = str(
                payload
            )

        return self._call(
            "publish",
            topic,
            payload,
            qos=qos,
            retain=retain,
        )

    # ========================================================
    # SUBSCRIBE
    # ========================================================

    def subscribe(
        self,
        topic: str,
        callback: Callable[
            [str, Any],
            None,
        ] | None = None,
        *,
        qos: int = 0,
    ) -> Any:
        """Subscribe to an MQTT topic."""

        self.ensure_connected()

        if callback is None:
            return self._call(
                "subscribe",
                topic,
                qos=qos,
            )

        def handler(
            client: Any,
            userdata: Any,
            message: Any,
        ) -> None:
            try:
                payload = self._decode_payload(
                    message.payload
                )

                callback(
                    message.topic,
                    payload,
                )

            except Exception:
                return

        return self._call(
            "subscribe",
            topic,
            qos=qos,
            callback=handler,
        )

    # ========================================================
    # DEVICE DISCOVERY
    # ========================================================

    def list_devices(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Return devices known through retained state.

        MQTT itself does not require a discovery protocol, so RENIX
        maintains a normalized local registry based on received state.
        """

        with self._lock:
            devices = set(
                self._states.keys()
            )

            devices.update(
                self._availability.keys()
            )

        return [
            self.get_device(
                device_id
            )
            for device_id in sorted(
                devices
            )
        ]

    def get_device(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return normalized MQTT device information."""

        device_id = str(
            device_id
        )

        state = self.get_state(
            device_id
        )

        return {
            "id": device_id,
            "device_id": device_id,
            "name": state.get(
                "name",
                device_id,
            ),
            "available": self._availability.get(
                device_id,
                True,
            ),
            "state": state,
            "protocol": "mqtt",
        }

    def get_state(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return cached device state."""

        with self._lock:
            return dict(
                self._states.get(
                    str(device_id),
                    {},
                )
            )

    # ========================================================
    # COMMANDS
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
        """Send a generic service command through MQTT."""

        target = (
            device_id
            or entity_id
        )

        if not target:
            raise ValueError(
                "device_id or entity_id is required."
            )

        payload = {
            "service": service,
            "device_id": target,
            "data": data or {},
        }

        payload.update(
            kwargs
        )

        return self.publish(
            self.command_topic(
                target
            ),
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
    # SWITCHES
    # ========================================================

    def turn_switch_on(
        self,
        device_id: str,
        **kwargs: Any,
    ) -> Any:
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
    # CLIMATE
    # ========================================================

    def set_temperature(
        self,
        device_id: str,
        temperature: float,
        **kwargs: Any,
    ) -> Any:
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
        mode = mode.strip().lower()

        return self.call_service(
            "climate.set_mode",
            device_id=device_id,
            data={
                "mode": mode,
                **kwargs,
            },
        )

    # ========================================================
    # SCENES
    # ========================================================

    def activate_scene(
        self,
        scene_id: str,
        **kwargs: Any,
    ) -> Any:
        return self.call_service(
            "scene.activate",
            entity_id=scene_id,
            data=kwargs,
        )

    # ========================================================
    # STATE HANDLING
    # ========================================================

    def update_state(
        self,
        device_id: str,
        state: dict[str, Any],
    ) -> None:
        """Update cached state and notify subscribers."""

        device_id = str(
            device_id
        )

        normalized = dict(
            state
        )

        with self._lock:
            self._states[
                device_id
            ] = normalized

        for callback in list(
            self._state_callbacks
        ):
            try:
                callback(
                    device_id,
                    dict(normalized),
                )
            except Exception:
                continue

    def update_availability(
        self,
        device_id: str,
        available: bool,
    ) -> None:
        """Update device availability."""

        with self._lock:
            self._availability[
                str(device_id)
            ] = bool(
                available
            )

    def add_state_callback(
        self,
        callback: Callable[
            [str, dict[str, Any]],
            None,
        ],
    ) -> None:
        """Register a state callback."""

        if not callable(callback):
            raise TypeError(
                "callback must be callable."
            )

        if callback not in self._state_callbacks:
            self._state_callbacks.append(
                callback
            )

    def remove_state_callback(
        self,
        callback: Callable[
            [str, dict[str, Any]],
            None,
        ],
    ) -> None:
        """Remove a state callback."""

        if callback in self._state_callbacks:
            self._state_callbacks.remove(
                callback
            )

    # ========================================================
    # MQTT MESSAGE PROCESSING
    # ========================================================

    def handle_message(
        self,
        topic: str,
        payload: Any,
    ) -> None:
        """Process an incoming MQTT message."""

        topic = str(
            topic
        ).strip(
            "/"
        )

        parts = topic.split(
            "/"
        )

        base_parts = self.base_topic.split(
            "/"
        )

        if parts[: len(base_parts)] != base_parts:
            return

        remaining = parts[
            len(base_parts):
        ]

        if len(remaining) < 2:
            return

        device_id = remaining[0]
        message_type = remaining[1]

        decoded = self._decode_payload(
            payload
        )

        if message_type == "state":
            if isinstance(
                decoded,
                dict,
            ):
                self.update_state(
                    device_id,
                    decoded,
                )

        elif message_type == "availability":
            self.update_availability(
                device_id,
                self._parse_availability(
                    decoded
                ),
            )

    # ========================================================
    # SUBSCRIPTIONS
    # ========================================================

    def _subscribe_to_base_topics(
        self,
    ) -> None:
        """Subscribe to device state and availability."""

        try:
            self.subscribe(
                f"{self.base_topic}/+/state",
                callback=self.handle_message,
            )

            self.subscribe(
                f"{self.base_topic}/+/availability",
                callback=self.handle_message,
            )

        except Exception:
            # Connection should remain usable even if the MQTT
            # implementation handles subscriptions separately.
            pass

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(self) -> bool:
        """Check MQTT provider health."""

        try:
            if not self.is_connected():
                return False

            client = self.client

            if client is None:
                return False

            connected = getattr(
                client,
                "is_connected",
                None,
            )

            if callable(connected):
                return bool(
                    connected()
                )

            return True

        except Exception:
            return False

    def capabilities(
        self,
    ) -> dict[str, bool]:
        """Return MQTT provider capabilities."""

        return {
            "devices": True,
            "state": True,
            "service_calls": True,
            "lights": True,
            "switches": True,
            "plugs": True,
            "fans": True,
            "climate": True,
            "scenes": True,
            "mqtt": True,
        }

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _decode_payload(
        payload: Any,
    ) -> Any:
        """Decode MQTT bytes/string/JSON payload."""

        if isinstance(
            payload,
            bytes,
        ):
            payload = payload.decode(
                "utf-8",
                errors="replace",
            )

        if not isinstance(
            payload,
            str,
        ):
            return payload

        value = payload.strip()

        if not value:
            return ""

        try:
            return json.loads(
                value
            )
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _parse_availability(
        payload: Any,
    ) -> bool:
        """Convert common availability values to bool."""

        if isinstance(
            payload,
            bool,
        ):
            return payload

        value = str(
            payload
        ).strip().lower()

        return value in {
            "online",
            "available",
            "true",
            "1",
            "on",
        }

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
    "MQTTProvider",
]


