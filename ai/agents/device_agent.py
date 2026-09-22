"""
RENIX AI - Device Agent

Handles connected-device operations.

Responsibilities:
- Discover devices
- Get device information
- Pair/connect/disconnect devices
- Send commands to supported devices
- Monitor device state
- Control smart-home devices
- Work with Bluetooth/Wi-Fi/MQTT/HTTP/WebSocket integrations
- Provide a safe abstraction layer for RENIX device management

This agent is intentionally modular. Device-specific functionality is delegated
to the `devices/` subsystem when those modules are available.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence


class DeviceAgent:
    """
    RENIX device-management agent.

    The agent does not assume that every device protocol is installed.
    It dynamically connects to the device subsystem when available and
    otherwise provides a controlled fallback implementation.
    """

    agent_name = "device_agent"
    agent_version = "1.0.0"

    agent_description = (
        "Discovers, connects, monitors and controls supported devices "
        "through the RENIX device subsystem."
    )

    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        priority: Any = None,
        logger: Optional[logging.Logger] = None,
        event_callback: Any = None,
        confirmation_callback: Any = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.agent_id = (
            agent_id
            or "device-agent"
        )

        self.name = (
            name
            or "RENIX Device Agent"
        )

        self.description = (
            description
            or self.agent_description
        )

        self.priority = priority

        self.logger = (
            logger
            or logging.getLogger(
                __name__
            )
        )

        self.event_callback = event_callback
        self.confirmation_callback = (
            confirmation_callback
        )

        self.context = (
            context.copy()
            if context
            else {}
        )

        self.enabled = True
        self.initialized = False

        self.device_manager = None
        self.discovery = None
        self.pairing = None

        self.bluetooth = None
        self.wifi = None
        self.mqtt = None
        self.http = None
        self.websocket = None

        self.smart_home_modules: Dict[
            str,
            Any,
        ] = {}

        self.mobile_modules: Dict[
            str,
            Any,
        ] = {}

        self.watch_modules: Dict[
            str,
            Any,
        ] = {}

        self.tv_modules: Dict[
            str,
            Any,
        ] = {}

        self._modules_loaded = False

        self._devices: Dict[
            str,
            Dict[str, Any],
        ] = {}

        self._connections: Dict[
            str,
            Dict[str, Any],
        ] = {}

        self._command_history: List[
            Dict[str, Any]
        ] = []

    # ======================================================================
    # INITIALIZATION
    # ======================================================================

    async def initialize(self) -> bool:
        """
        Initialize the device subsystem.
        """

        if self.initialized:
            return True

        self._load_modules()

        self.initialized = True

        await self._emit_event(
            "device_agent_initialized",
            {
                "agent_id": self.agent_id,
            },
        )

        return True

    async def shutdown(self) -> bool:
        """
        Disconnect active devices and shut down the agent.
        """

        device_ids = list(
            self._connections.keys()
        )

        for device_id in device_ids:

            try:
                await self.disconnect_device(
                    device_id
                )
            except Exception:
                self.logger.exception(
                    "Failed to disconnect device %s",
                    device_id,
                )

        self.initialized = False

        await self._emit_event(
            "device_agent_shutdown",
            {
                "agent_id": self.agent_id,
            },
        )

        return True

    def _load_modules(self) -> None:
        """
        Dynamically load modules from the RENIX devices package.

        Missing modules are allowed because the project is modular and
        some protocols/devices may not be installed yet.
        """

        if self._modules_loaded:
            return

        self._modules_loaded = True

        core_modules = {
            "device_manager": (
                "devices.device_manager",
                "DeviceManager",
            ),
            "discovery": (
                "devices.discovery",
                "DeviceDiscovery",
            ),
            "pairing": (
                "devices.pairing",
                "DevicePairing",
            ),
        }

        for attribute, (
            module_name,
            class_name,
        ) in core_modules.items():

            instance = self._load_class(
                module_name,
                class_name,
            )

            if instance is not None:

                setattr(
                    self,
                    attribute,
                    instance,
                )

        protocol_modules = {
            "bluetooth": (
                "devices.protocols.bluetooth",
                "BluetoothProtocol",
            ),
            "wifi": (
                "devices.protocols.wifi",
                "WiFiProtocol",
            ),
            "mqtt": (
                "devices.protocols.mqtt",
                "MQTTProtocol",
            ),
            "http": (
                "devices.protocols.http",
                "HTTPProtocol",
            ),
            "websocket": (
                "devices.protocols.websocket",
                "WebSocketProtocol",
            ),
        }

        for attribute, (
            module_name,
            class_name,
        ) in protocol_modules.items():

            instance = self._load_class(
                module_name,
                class_name,
            )

            if instance is not None:

                setattr(
                    self,
                    attribute,
                    instance,
                )

        smart_home = {
            "lights": (
                "devices.smart_home.lights",
                "LightsController",
            ),
            "fans": (
                "devices.smart_home.fans",
                "FansController",
            ),
            "ac": (
                "devices.smart_home.ac",
                "ACController",
            ),
            "tv": (
                "devices.smart_home.tv",
                "TVController",
            ),
            "speakers": (
                "devices.smart_home.speakers",
                "SpeakerController",
            ),
            "plugs": (
                "devices.smart_home.plugs",
                "PlugController",
            ),
            "cameras": (
                "devices.smart_home.cameras",
                "CameraController",
            ),
            "sensors": (
                "devices.smart_home.sensors",
                "SensorController",
            ),
        }

        for key, (
            module_name,
            class_name,
        ) in smart_home.items():

            instance = self._load_class(
                module_name,
                class_name,
            )

            if instance is not None:

                self.smart_home_modules[
                    key
                ] = instance

        self._load_optional_group(
            "mobile",
            self.mobile_modules,
        )

        self._load_optional_group(
            "smartwatch",
            self.watch_modules,
        )

        self._load_optional_group(
            "tv",
            self.tv_modules,
        )

    def _load_optional_group(
        self,
        group_name: str,
        target: Dict[str, Any],
    ) -> None:
        """
        Load optional device modules from a package.
        """

        module_name = (
            f"devices.{group_name}"
        )

        try:

            module = __import__(
                module_name,
                fromlist=["*"],
            )

        except Exception as exc:

            self.logger.debug(
                "Optional device group unavailable: %s (%s)",
                group_name,
                exc,
            )

            return

        for attribute_name in dir(
            module
        ):

            if attribute_name.startswith(
                "_"
            ):
                continue

            candidate = getattr(
                module,
                attribute_name,
            )

            if isinstance(
                candidate,
                type,
            ):

                try:
                    target[
                        attribute_name
                    ] = candidate()
                except Exception:
                    target[
                        attribute_name
                    ] = candidate

    @staticmethod
    def _load_class(
        module_name: str,
        class_name: str,
    ) -> Any:
        """
        Import a class and instantiate it.
        """

        try:

            module = __import__(
                module_name,
                fromlist=[
                    class_name
                ],
            )

            cls = getattr(
                module,
                class_name,
            )

            try:
                return cls()
            except TypeError:
                return cls

        except Exception:

            return None

    # ======================================================================
    # MAIN ROUTER
    # ======================================================================

    async def execute(
        self,
        instruction: Optional[str] = None,
        *,
        action: Optional[str] = None,
        parameters: Optional[
            Dict[str, Any]
        ] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute a device-related instruction.

        Examples:

            await agent.execute(
                action="discover"
            )

            await agent.execute(
                action="connect",
                parameters={
                    "device_id": "abc"
                }
            )

            await agent.execute(
                action="command",
                parameters={
                    "device_id": "abc",
                    "command": "status"
                }
            )
        """

        if not self.initialized:
            await self.initialize()

        params = dict(
            parameters
            or {}
        )

        params.update(
            kwargs
        )

        resolved_action = (
            action
            or params.pop(
                "action",
                None,
            )
            or self._resolve_action(
                instruction
                or ""
            )
        )

        resolved_action = (
            str(
                resolved_action
            )
            .strip()
            .lower()
        )

        handlers = {
            "device": self.device_info,
            "info": self.device_info,
            "status": self.device_status,
            "device_status": self.device_status,
            "discover": self.discover_devices,
            "scan": self.discover_devices,
            "find": self.discover_devices,
            "pair": self.pair_device,
            "connect": self.connect_device,
            "disconnect": self.disconnect_device,
            "command": self.send_command,
            "control": self.send_command,
            "devices": self.list_devices,
            "list": self.list_devices,
            "connected": self.list_connected_devices,
            "smart_home": self.smart_home_control,
            "light": self.control_lights,
            "lights": self.control_lights,
            "fan": self.control_fans,
            "fans": self.control_fans,
            "ac": self.control_ac,
            "tv": self.control_tv,
            "speaker": self.control_speakers,
            "speakers": self.control_speakers,
            "plug": self.control_plugs,
            "plugs": self.control_plugs,
            "camera": self.control_cameras,
            "cameras": self.control_cameras,
            "sensor": self.sensor_status,
            "sensors": self.sensor_status,
        }

        handler = handlers.get(
            resolved_action
        )

        if handler is None:

            return self._failure(
                (
                    f"Unknown device action: "
                    f"{resolved_action}"
                )
            )

        try:

            result = handler(
                **params
            )

            if asyncio.iscoroutine(
                result
            ):
                result = await result

            return result

        except Exception as exc:

            self.logger.exception(
                "Device action failed: %s",
                resolved_action,
            )

            return self._failure(
                str(exc)
            )

    def _resolve_action(
        self,
        instruction: str,
    ) -> str:

        text = (
            instruction
            .strip()
            .lower()
        )

        if any(
            word in text
            for word in (
                "discover",
                "scan devices",
                "find devices",
                "nearby devices",
            )
        ):
            return "discover"

        if "pair" in text:
            return "pair"

        if "disconnect" in text:
            return "disconnect"

        if "connect" in text:
            return "connect"

        if any(
            word in text
            for word in (
                "smart home",
                "smart-home",
            )
        ):
            return "smart_home"

        if "light" in text:
            return "lights"

        if "fan" in text:
            return "fans"

        if (
            "air conditioner" in text
            or "ac " in text
            or text == "ac"
        ):
            return "ac"

        if "television" in text or " tv" in text:
            return "tv"

        if "speaker" in text:
            return "speakers"

        if "plug" in text:
            return "plugs"

        if "camera" in text:
            return "cameras"

        if "sensor" in text:
            return "sensors"

        if "status" in text:
            return "status"

        if (
            "control" in text
            or "turn on" in text
            or "turn off" in text
            or "set " in text
        ):
            return "command"

        return "device"

    # ======================================================================
    # DEVICE DISCOVERY
    # ======================================================================

    async def discover_devices(
        self,
        protocol: Optional[str] = None,
        device_type: Optional[str] = None,
        timeout: float = 5.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Discover nearby/available devices.
        """

        discovered: List[
            Dict[str, Any]
        ] = []

        if self.discovery is not None:

            result = await self._call_module(
                self.discovery,
                (
                    "discover",
                    "scan",
                    "find_devices",
                ),
                protocol=protocol,
                device_type=device_type,
                timeout=timeout,
                **kwargs,
            )

            if result is not None:

                discovered = self._normalize_devices(
                    result
                )

        if not discovered and self.device_manager is not None:

            result = await self._call_module(
                self.device_manager,
                (
                    "discover",
                    "scan",
                    "list_available",
                ),
                protocol=protocol,
                device_type=device_type,
                timeout=timeout,
                **kwargs,
            )

            if result is not None:

                discovered = self._normalize_devices(
                    result
                )

        for device in discovered:

            device_id = self._device_id(
                device
            )

            self._devices[
                device_id
            ] = device

        await self._emit_event(
            "devices_discovered",
            {
                "count": len(
                    discovered
                ),
                "protocol": protocol,
                "device_type": device_type,
            },
        )

        return {
            "success": True,
            "action": "discover",
            "count": len(
                discovered
            ),
            "devices": discovered,
        }

    # ======================================================================
    # DEVICE LISTING
    # ======================================================================

    async def list_devices(
        self,
        connected_only: bool = False,
        device_type: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        List known devices.
        """

        devices = list(
            self._devices.values()
        )

        if connected_only:

            devices = [
                device
                for device in devices
                if self._device_id(
                    device
                )
                in self._connections
            ]

        if device_type:

            normalized_type = (
                device_type.lower()
            )

            devices = [
                device
                for device in devices
                if str(
                    device.get(
                        "type",
                        "",
                    )
                ).lower()
                == normalized_type
            ]

        return {
            "success": True,
            "action": "list_devices",
            "count": len(
                devices
            ),
            "devices": devices,
        }

    async def list_connected_devices(
        self,
        **_: Any,
    ) -> Dict[str, Any]:

        return await self.list_devices(
            connected_only=True
        )

    # ======================================================================
    # PAIRING
    # ======================================================================

    async def pair_device(
        self,
        device_id: Optional[str] = None,
        pin: Optional[str] = None,
        protocol: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Pair a device.
        """

        if not device_id:

            return self._failure(
                "device_id is required"
            )

        device = self._devices.get(
            device_id,
            {
                "id": device_id,
                "name": device_id,
                "protocol": protocol,
            },
        )

        if self.pairing is not None:

            result = await self._call_module(
                self.pairing,
                (
                    "pair",
                    "pair_device",
                    "authenticate",
                ),
                device=device,
                device_id=device_id,
                pin=pin,
                protocol=protocol,
                **kwargs,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "pair",
                    "device_id": device_id,
                    "result": result,
                }

        device[
            "paired"
        ] = True

        self._devices[
            device_id
        ] = device

        await self._emit_event(
            "device_paired",
            {
                "device_id": device_id,
            },
        )

        return {
            "success": True,
            "action": "pair",
            "device_id": device_id,
            "paired": True,
            "method": "fallback",
        }

    # ======================================================================
    # CONNECTION
    # ======================================================================

    async def connect_device(
        self,
        device_id: Optional[str] = None,
        protocol: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Connect to a device.
        """

        if not device_id:

            return self._failure(
                "device_id is required"
            )

        device = self._devices.get(
            device_id,
            {
                "id": device_id,
                "name": device_id,
                "protocol": protocol,
            },
        )

        if self.device_manager is not None:

            result = await self._call_module(
                self.device_manager,
                (
                    "connect",
                    "connect_device",
                ),
                device=device,
                device_id=device_id,
                protocol=protocol,
                **kwargs,
            )

            if result is not None:

                self._connections[
                    device_id
                ] = {
                    "device_id": device_id,
                    "connected_at": self._timestamp(),
                    "result": result,
                }

                return {
                    "success": True,
                    "action": "connect",
                    "device_id": device_id,
                    "result": result,
                }

        device[
            "connected"
        ] = True

        self._devices[
            device_id
        ] = device

        self._connections[
            device_id
        ] = {
            "device_id": device_id,
            "connected_at": self._timestamp(),
            "protocol": (
                protocol
                or device.get(
                    "protocol"
                )
            ),
        }

        await self._emit_event(
            "device_connected",
            {
                "device_id": device_id,
            },
        )

        return {
            "success": True,
            "action": "connect",
            "device_id": device_id,
            "connected": True,
            "method": "fallback",
        }

    async def disconnect_device(
        self,
        device_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Disconnect from a device.
        """

        if not device_id:

            return self._failure(
                "device_id is required"
            )

        device = self._devices.get(
            device_id
        )

        if self.device_manager is not None:

            result = await self._call_module(
                self.device_manager,
                (
                    "disconnect",
                    "disconnect_device",
                ),
                device=device,
                device_id=device_id,
                **kwargs,
            )

            if result is not None:

                self._connections.pop(
                    device_id,
                    None,
                )

                return {
                    "success": True,
                    "action": "disconnect",
                    "device_id": device_id,
                    "result": result,
                }

        if device is not None:

            device[
                "connected"
            ] = False

        self._connections.pop(
            device_id,
            None,
        )

        await self._emit_event(
            "device_disconnected",
            {
                "device_id": device_id,
            },
        )

        return {
            "success": True,
            "action": "disconnect",
            "device_id": device_id,
            "connected": False,
            "method": "fallback",
        }

    # ======================================================================
    # DEVICE COMMANDS
    # ======================================================================

    async def send_command(
        self,
        device_id: Optional[str] = None,
        command: Optional[str] = None,
        value: Any = None,
        protocol: Optional[str] = None,
        require_confirmation: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Send a command to a device.

        Potentially destructive commands can request confirmation through
        `confirmation_callback`.
        """

        if not device_id:

            return self._failure(
                "device_id is required"
            )

        if not command:

            return self._failure(
                "command is required"
            )

        if require_confirmation:

            confirmed = await self._request_confirmation(
                {
                    "device_id": device_id,
                    "command": command,
                    "value": value,
                }
            )

            if not confirmed:

                return {
                    "success": False,
                    "action": "command",
                    "device_id": device_id,
                    "command": command,
                    "cancelled": True,
                }

        device = self._devices.get(
            device_id,
            {
                "id": device_id,
                "name": device_id,
                "protocol": protocol,
            },
        )

        protocol_name = (
            protocol
            or device.get(
                "protocol"
            )
        )

        protocol_object = (
            self._get_protocol(
                protocol_name
            )
        )

        result = None

        if protocol_object is not None:

            result = await self._call_module(
                protocol_object,
                (
                    "send_command",
                    "send",
                    "execute",
                    "command",
                    "control",
                ),
                device=device,
                device_id=device_id,
                command=command,
                value=value,
                **kwargs,
            )

        if result is None and self.device_manager is not None:

            result = await self._call_module(
                self.device_manager,
                (
                    "send_command",
                    "command",
                    "control",
                    "execute",
                ),
                device=device,
                device_id=device_id,
                command=command,
                value=value,
                **kwargs,
            )

        record = {
            "timestamp": self._timestamp(),
            "device_id": device_id,
            "command": command,
            "value": value,
            "protocol": protocol_name,
        }

        self._command_history.append(
            record
        )

        await self._emit_event(
            "device_command_sent",
            record,
        )

        return {
            "success": True,
            "action": "command",
            "device_id": device_id,
            "command": command,
            "value": value,
            "protocol": protocol_name,
            "result": result,
            "method": (
                "protocol"
                if protocol_object is not None
                else "device_manager"
                if self.device_manager is not None
                else "fallback"
            ),
        }

    # ======================================================================
    # DEVICE STATUS
    # ======================================================================

    async def device_status(
        self,
        device_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Get device status.
        """

        if not device_id:

            return self._failure(
                "device_id is required"
            )

        device = self._devices.get(
            device_id
        )

        if self.device_manager is not None:

            result = await self._call_module(
                self.device_manager,
                (
                    "status",
                    "get_status",
                    "device_status",
                ),
                device_id=device_id,
                device=device,
                **kwargs,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "status",
                    "device_id": device_id,
                    "status": result,
                }

        if device is None:

            return {
                "success": False,
                "action": "status",
                "device_id": device_id,
                "error": "Device is not known.",
            }

        return {
            "success": True,
            "action": "status",
            "device_id": device_id,
            "status": {
                **device,
                "connected": (
                    device_id
                    in self._connections
                ),
            },
        }

    async def device_info(
        self,
        device_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Return device information.
        """

        if device_id:

            device = self._devices.get(
                device_id
            )

            if device is None:

                return {
                    "success": False,
                    "error": "Device not found.",
                    "device_id": device_id,
                }

            return {
                "success": True,
                "action": "device_info",
                "device": device,
                "connection": self._connections.get(
                    device_id
                ),
            }

        return await self.list_devices(
            **kwargs
        )

    # ======================================================================
    # SMART HOME
    # ======================================================================

    async def smart_home_control(
        self,
        category: Optional[str] = None,
        device_id: Optional[str] = None,
        command: Optional[str] = None,
        value: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Route smart-home commands.
        """

        if not category:

            return self._failure(
                "category is required"
            )

        category_name = (
            category
            .strip()
            .lower()
        )

        handler_map = {
            "light": self.control_lights,
            "lights": self.control_lights,
            "fan": self.control_fans,
            "fans": self.control_fans,
            "ac": self.control_ac,
            "tv": self.control_tv,
            "speaker": self.control_speakers,
            "speakers": self.control_speakers,
            "plug": self.control_plugs,
            "plugs": self.control_plugs,
            "camera": self.control_cameras,
            "cameras": self.control_cameras,
            "sensor": self.sensor_status,
            "sensors": self.sensor_status,
        }

        handler = handler_map.get(
            category_name
        )

        if handler is None:

            return self._failure(
                f"Unsupported smart-home category: {category}"
            )

        return await handler(
            device_id=device_id,
            command=command,
            value=value,
            **kwargs,
        )

    async def control_lights(
        self,
        device_id: Optional[str] = None,
        command: Optional[str] = None,
        value: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        return await self._smart_home_command(
            "lights",
            device_id,
            command,
            value,
            **kwargs,
        )

    async def control_fans(
        self,
        device_id: Optional[str] = None,
        command: Optional[str] = None,
        value: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        return await self._smart_home_command(
            "fans",
            device_id,
            command,
            value,
            **kwargs,
        )

    async def control_ac(
        self,
        device_id: Optional[str] = None,
        command: Optional[str] = None,
        value: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        return await self._smart_home_command(
            "ac",
            device_id,
            command,
            value,
            **kwargs,
        )

    async def control_tv(
        self,
        device_id: Optional[str] = None,
        command: Optional[str] = None,
        value: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        return await self._smart_home_command(
            "tv",
            device_id,
            command,
            value,
            **kwargs,
        )

    async def control_speakers(
        self,
        device_id: Optional[str] = None,
        command: Optional[str] = None,
        value: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        return await self._smart_home_command(
            "speakers",
            device_id,
            command,
            value,
            **kwargs,
        )

    async def control_plugs(
        self,
        device_id: Optional[str] = None,
        command: Optional[str] = None,
        value: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        return await self._smart_home_command(
            "plugs",
            device_id,
            command,
            value,
            **kwargs,
        )

    async def control_cameras(
        self,
        device_id: Optional[str] = None,
        command: Optional[str] = None,
        value: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        return await self._smart_home_command(
            "cameras",
            device_id,
            command,
            value,
            **kwargs,
        )

    async def sensor_status(
        self,
        device_id: Optional[str] = None,
        command: Optional[str] = None,
        value: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        return await self._smart_home_command(
            "sensors",
            device_id,
            command or "status",
            value,
            **kwargs,
        )

    async def _smart_home_command(
        self,
        category: str,
        device_id: Optional[str],
        command: Optional[str],
        value: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute a smart-home command through the relevant controller.
        """

        controller = self.smart_home_modules.get(
            category
        )

        if controller is not None:

            result = await self._call_module(
                controller,
                (
                    "control",
                    "command",
                    "execute",
                    "set",
                ),
                device_id=device_id,
                command=command,
                value=value,
                **kwargs,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": category,
                    "device_id": device_id,
                    "command": command,
                    "value": value,
                    "result": result,
                }

        if not device_id:

            return self._failure(
                "device_id is required"
            )

        return await self.send_command(
            device_id=device_id,
            command=command,
            value=value,
            **kwargs,
        )

    # ======================================================================
    # PROTOCOL HANDLING
    # ======================================================================

    def _get_protocol(
        self,
        protocol: Optional[str],
    ) -> Any:

        if not protocol:
            return None

        normalized = (
            str(
                protocol
            )
            .strip()
            .lower()
        )

        aliases = {
            "ble": "bluetooth",
            "bt": "bluetooth",
            "wi-fi": "wifi",
            "web-socket": "websocket",
        }

        normalized = aliases.get(
            normalized,
            normalized,
        )

        return getattr(
            self,
            normalized,
            None,
        )

    # ======================================================================
    # CALLBACKS
    # ======================================================================

    async def _request_confirmation(
        self,
        payload: Dict[str, Any],
    ) -> bool:
        """
        Request user confirmation for a sensitive device command.
        """

        callback = (
            self.confirmation_callback
        )

        if callback is None:

            # Safe default: do not execute commands requiring explicit
            # confirmation if no confirmation system is connected.
            return False

        try:

            result = callback(
                payload
            )

            if asyncio.iscoroutine(
                result
            ):
                result = await result

            return bool(
                result
            )

        except Exception as exc:

            self.logger.exception(
                "Confirmation callback failed: %s",
                exc,
            )

            return False

    async def _emit_event(
        self,
        event_name: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        Send an event to the RENIX event system if connected.
        """

        if self.event_callback is None:
            return

        try:

            result = self.event_callback(
                event_name,
                payload,
            )

            if asyncio.iscoroutine(
                result
            ):
                await result

        except Exception:

            self.logger.exception(
                "Device event callback failed."
            )

    # ======================================================================
    # MODULE CALLING
    # ======================================================================

    async def _call_module(
        self,
        module: Any,
        method_names: Iterable[str],
        **kwargs: Any,
    ) -> Any:
        """
        Call the first compatible method on a subsystem module.
        """

        for method_name in method_names:

            method = getattr(
                module,
                method_name,
                None,
            )

            if not callable(
                method
            ):
                continue

            try:

                result = method(
                    **kwargs
                )

                if asyncio.iscoroutine(
                    result
                ):
                    result = await result

                return result

            except TypeError:

                try:

                    result = method(
                        kwargs
                    )

                    if asyncio.iscoroutine(
                        result
                    ):
                        result = await result

                    return result

                except Exception as exc:

                    self.logger.debug(
                        "Device method %s failed: %s",
                        method_name,
                        exc,
                    )

            except Exception as exc:

                self.logger.debug(
                    "Device method %s failed: %s",
                    method_name,
                    exc,
                )

        return None

    # ======================================================================
    # NORMALIZATION
    # ======================================================================

    @staticmethod
    def _normalize_devices(
        result: Any,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Convert different discovery-result formats into a consistent list.
        """

        if result is None:
            return []

        if isinstance(
            result,
            dict,
        ):

            if isinstance(
                result.get(
                    "devices"
                ),
                Sequence,
            ):

                result = result[
                    "devices"
                ]

            else:

                result = [
                    result
                ]

        if not isinstance(
            result,
            Sequence,
        ):

            return []

        normalized = []

        for item in result:

            if isinstance(
                item,
                dict,
            ):

                normalized.append(
                    dict(
                        item
                    )
                )

            else:

                normalized.append(
                    {
                        "id": str(
                            item
                        ),
                        "name": str(
                            item
                        ),
                    }
                )

        return normalized

    @staticmethod
    def _device_id(
        device: Dict[str, Any],
    ) -> str:

        value = (
            device.get("id")
            or device.get("device_id")
            or device.get("uuid")
            or device.get("address")
            or device.get("name")
        )

        if value is None:

            value = (
                f"device-"
                f"{abs(hash(str(device)))}"
            )

        return str(
            value
        )

    # ======================================================================
    # HELPERS
    # ======================================================================

    @staticmethod
    def _timestamp() -> str:

        return (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

    @staticmethod
    def _failure(
        message: str,
    ) -> Dict[str, Any]:

        return {
            "success": False,
            "error": message,
        }

    # ======================================================================
    # INFORMATION
    # ======================================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:

        return {
            "agent": self.agent_name,
            "agent_id": self.agent_id,
            "version": self.agent_version,
            "enabled": self.enabled,
            "initialized": self.initialized,
            "known_devices": len(
                self._devices
            ),
            "connected_devices": len(
                self._connections
            ),
            "commands_sent": len(
                self._command_history
            ),
            "loaded_protocols": [
                name
                for name in (
                    "bluetooth",
                    "wifi",
                    "mqtt",
                    "http",
                    "websocket",
                )
                if getattr(
                    self,
                    name,
                    None,
                )
                is not None
            ],
        }


__all__ = [
    "DeviceAgent",
]


