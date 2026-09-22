"""
RENIX Integrations
==================

Central integration subsystem for RENIX.

Provides a unified interface for connecting RENIX with
external services, APIs, webhooks, and provider modules.
"""

from .api_manager import APIManager
from .authentication import AuthenticationManager
from .services import ServiceManager, Service
from .webhooks import WebhookManager, Webhook

__all__ = [
    "APIManager",
    "AuthenticationManager",
    "ServiceManager",
    "Service",
    "WebhookManager",
    "Webhook",
]

__version__ = "1.0.0"


