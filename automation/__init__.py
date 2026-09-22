"""
RENIX Automation Package
========================

Central automation package for RENIX.

Provides:
    - Automation engine
    - Workflow execution
    - Action execution
    - Trigger management
    - Routine management
    - Scheduled tasks
    - Macros
"""

from .automation_engine import AutomationEngine
from .workflow_engine import WorkflowEngine
from .action_executor import ActionExecutor
from .trigger_manager import TriggerManager
from .routine_manager import RoutineManager
from .scheduled_tasks import ScheduledTaskManager
from .macros import MacroManager

__all__ = [
    "AutomationEngine",
    "WorkflowEngine",
    "ActionExecutor",
    "TriggerManager",
    "RoutineManager",
    "ScheduledTaskManager",
    "MacroManager",
]


