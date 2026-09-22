"""
RENIX AI Agents Package

Central package exports for all RENIX autonomous agents.

Agents available in this package:

- BaseAgent
- ComputerAgent
- FileAgent
- BrowserAgent
- CodingAgent
- ResearchAgent
- StudyAgent
- CricketAgent
- DeviceAgent
- RoboticsAgent
- SystemAgent
- AutomationAgent

This file contains no agent implementation logic.
It only exposes the agents through a clean package API.
"""

from __future__ import annotations

# ============================================================================
# BASE AGENT
# ============================================================================

from .base_agent import BaseAgent


# ============================================================================
# COMPUTER AGENT
# ============================================================================

from .computer_agent import ComputerAgent


# ============================================================================
# FILE AGENT
# ============================================================================

from .file_agent import FileAgent


# ============================================================================
# BROWSER AGENT
# ============================================================================

from .browser_agent import BrowserAgent


# ============================================================================
# CODING AGENT
# ============================================================================

from .coding_agent import CodingAgent


# ============================================================================
# RESEARCH AGENT
# ============================================================================

from .research_agent import ResearchAgent


# ============================================================================
# STUDY AGENT
# ============================================================================

from .study_agent import StudyAgent


# ============================================================================
# CRICKET AGENT
# ============================================================================

from .cricket_agent import CricketAgent


# ============================================================================
# DEVICE AGENT
# ============================================================================

from .device_agent import DeviceAgent


# ============================================================================
# ROBOTICS AGENT
# ============================================================================

from .robotics_agent import RoboticsAgent


# ============================================================================
# SYSTEM AGENT
# ============================================================================

from .system_agent import SystemAgent


# ============================================================================
# AUTOMATION AGENT
# ============================================================================

from .automation_agent import AutomationAgent


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "BaseAgent",
    "ComputerAgent",
    "FileAgent",
    "BrowserAgent",
    "CodingAgent",
    "ResearchAgent",
    "StudyAgent",
    "CricketAgent",
    "DeviceAgent",
    "RoboticsAgent",
    "SystemAgent",
    "AutomationAgent",
]


