"""
RENIX Holographic UI Widgets
"""

from .clock import ClockWidget
from .system_status import SystemStatusWidget
from .cpu import CPUWidget
from .gpu import GPUWidget
from .ram import RAMWidget
from .network import NetworkWidget
from .weather import WeatherWidget
from .calendar import CalendarWidget
from .tasks import TasksWidget
from .media import MediaWidget

__all__ = [
    "ClockWidget",
    "SystemStatusWidget",
    "CPUWidget",
    "GPUWidget",
    "RAMWidget",
    "NetworkWidget",
    "WeatherWidget",
    "CalendarWidget",
    "TasksWidget",
    "MediaWidget",
]


