"""
RENIX Smart Home Providers
==========================

Provider integrations for smart-home devices and services.

These providers are responsible for communicating with external
smart-home platforms and protocols. Device-level logic remains in
the RENIX devices/smart_home package.
"""

from .smart_home_provider import SmartHomeProvider
from .home_assistant import HomeAssistantProvider

__all__ = [
    "SmartHomeProvider",
    "HomeAssistantProvider",
]


