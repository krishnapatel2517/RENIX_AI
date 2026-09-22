"""
RENIX Notifications
===================

Notification subsystem for RENIX.

Provides:
- Desktop notifications
- Notification priorities
- Notification rules
- Notification management
"""

from .notification_manager import NotificationManager
from .desktop_notifications import DesktopNotificationManager
from .priority import (
    NotificationPriority,
    get_priority_value,
)
from .notification_rules import (
    NotificationRule,
    NotificationRuleEngine,
)

__all__ = [
    "NotificationManager",
    "DesktopNotificationManager",
    "NotificationPriority",
    "get_priority_value",
    "NotificationRule",
    "NotificationRuleEngine",
]

__version__ = "1.0.0"



