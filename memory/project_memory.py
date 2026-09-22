"""
RENIX Project Memory
====================

Project-focused long-term memory for RENIX.

Responsibilities:
    - Store project metadata
    - Track project goals
    - Track project decisions
    - Track milestones
    - Track files/modules
    - Track project tasks
    - Track project context
    - Track project activity
    - Track project status
    - Search project information
    - Export/import project memory

This module is a MEMORY layer.

It does NOT:
    - execute shell commands
    - edit files
    - run applications
    - browse the internet
    - call an LLM

Those responsibilities belong to RENIX's agents, execution layer,
coding system, browser system, and AI layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Iterable, Optional
import logging
import uuid


logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class ProjectStatus(str, Enum):
    """Lifecycle status of a project."""

    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ProjectPriority(str, Enum):
    """Priority assigned to a project."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ProjectItemType(str, Enum):
    """Types of objects stored inside project memory."""

    GOAL = "goal"
    DECISION = "decision"
    MILESTONE = "milestone"
    FILE = "file"
    TASK = "task"
    NOTE = "note"
    EVENT = "event"
    REQUIREMENT = "requirement"


class ProjectItemStatus(str, Enum):
    """Status of project-level items."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


# ============================================================================
# EXCEPTIONS
# ============================================================================


class ProjectMemoryError(Exception):
    """Base exception for project-memory failures."""


class ProjectNotFoundError(ProjectMemoryError):
    """Raised when a project does not exist."""


class ProjectValidationError(ProjectMemoryError):
    """Raised when project data is invalid."""


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ProjectGoal:
    """A project goal."""

    goal_id: str
    title: str
    description: str = ""
    status: ProjectItemStatus = ProjectItemStatus.OPEN
    priority: ProjectPriority = ProjectPriority.NORMAL
    progress: float = 0.0
    deadline: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ProjectDecision:
    """A decision made during project development."""

    decision_id: str
    title: str
    decision: str
    reasoning: str = ""
    alternatives: list[str] = field(default_factory=list)
    impact: str = ""
    decided_by: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ProjectMilestone:
    """A significant project milestone."""

    milestone_id: str
    title: str
    description: str = ""
    status: ProjectItemStatus = ProjectItemStatus.OPEN
    progress: float = 0.0
    deadline: Optional[str] = None
    completed_at: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ProjectFile:
    """A file/module associated with a project."""

    file_id: str
    path: str
    description: str = ""
    file_type: Optional[str] = None
    module: Optional[str] = None
    role: Optional[str] = None
    status: ProjectItemStatus = ProjectItemStatus.OPEN
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ProjectTask:
    """A task belonging to a project."""

    task_id: str
    title: str
    description: str = ""
    status: ProjectItemStatus = ProjectItemStatus.OPEN
    priority: ProjectPriority = ProjectPriority.NORMAL
    assigned_agent: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None


@dataclass
class ProjectNote:
    """Free-form project knowledge."""

    note_id: str
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ProjectEvent:
    """An event/activity entry in project history."""

    event_id: str
    event_type: str
    description: str
    actor: Optional[str] = None
    related_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ProjectRequirement:
    """A requirement for a project."""

    requirement_id: str
    title: str
    description: str = ""
    status: ProjectItemStatus = ProjectItemStatus.OPEN
    priority: ProjectPriority = ProjectPriority.NORMAL
    source: Optional[str] = None
    acceptance_criteria: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class Project:
    """Complete project-memory record."""

    project_id: str
    name: str
    description: str = ""

    status: ProjectStatus = ProjectStatus.PLANNING
    priority: ProjectPriority = ProjectPriority.NORMAL

    owner_id: Optional[str] = None

    goals: list[ProjectGoal] = field(default_factory=list)
    decisions: list[ProjectDecision] = field(default_factory=list)
    milestones: list[ProjectMilestone] = field(default_factory=list)
    files: list[ProjectFile] = field(default_factory=list)
    tasks: list[ProjectTask] = field(default_factory=list)
    notes: list[ProjectNote] = field(default_factory=list)
    events: list[ProjectEvent] = field(default_factory=list)
    requirements: list[ProjectRequirement] = field(default_factory=list)

    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    last_activity_at: Optional[str] = None


# ============================================================================
# PROJECT MEMORY
# ============================================================================


class ProjectMemory:
    """
    Central project-memory service for RENIX.

    Example:

        project = memory.create_project(
            "RENIX AI",
            "Personal AI assistant project",
        )

        memory.add_goal(
            project.project_id,
            "Build holographic UI",
        )

        memory.add_file(
            project.project_id,
            "core/orchestrator.py",
            description="Main orchestration layer",
        )

    Other RENIX systems can query this memory to understand what a
    project is, what has already been built, what decisions were made,
    and what remains to be done.
    """

    def __init__(self) -> None:

        self._projects: dict[str, Project] = {}

        self._lock = RLock()

    # ========================================================================
    # PROJECT CREATION
    # ========================================================================

    def create_project(
        self,
        name: str,
        description: str = "",
        *,
        status: ProjectStatus = ProjectStatus.PLANNING,
        priority: ProjectPriority = ProjectPriority.NORMAL,
        owner_id: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Project:
        """Create a new project."""

        name = str(name).strip()

        if not name:
            raise ProjectValidationError(
                "Project name cannot be empty."
            )

        project = Project(
            project_id=self._new_id("project"),
            name=name,
            description=str(description).strip(),
            status=status,
            priority=priority,
            owner_id=owner_id,
            context=dict(context or {}),
            metadata=dict(metadata or {}),
        )

        with self._lock:

            self._projects[project.project_id] = project

        self._record_event(
            project,
            "project_created",
            f"Project '{project.name}' was created.",
        )

        return project

    # ========================================================================
    # PROJECT READ
    # ========================================================================

    def get_project(
        self,
        project_id: str,
    ) -> Optional[Project]:
        """Return a project by ID."""

        with self._lock:
            return self._projects.get(project_id)

    def require_project(
        self,
        project_id: str,
    ) -> Project:
        """Return project or raise ProjectNotFoundError."""

        project = self.get_project(project_id)

        if project is None:
            raise ProjectNotFoundError(
                f"Project not found: {project_id}"
            )

        return project

    def exists(
        self,
        project_id: str,
    ) -> bool:

        with self._lock:
            return project_id in self._projects

    # ========================================================================
    # PROJECT UPDATE
    # ========================================================================

    def update_project(
        self,
        project_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[ProjectStatus] = None,
        priority: Optional[ProjectPriority] = None,
        context: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Project:
        """Update project-level metadata."""

        with self._lock:

            project = self.require_project(project_id)

            if name is not None:
                name = str(name).strip()

                if not name:
                    raise ProjectValidationError(
                        "Project name cannot be empty."
                    )

                project.name = name

            if description is not None:
                project.description = str(description).strip()

            if status is not None:
                project.status = status

            if priority is not None:
                project.priority = priority

            if context is not None:
                project.context.update(context)

            if metadata is not None:
                project.metadata.update(metadata)

            self._touch(project)

            return project

    # ========================================================================
    # GOALS
    # ========================================================================

    def add_goal(
        self,
        project_id: str,
        title: str,
        description: str = "",
        *,
        priority: ProjectPriority = ProjectPriority.NORMAL,
        deadline: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProjectGoal:
        """Add a goal to a project."""

        title = str(title).strip()

        if not title:
            raise ProjectValidationError(
                "Goal title cannot be empty."
            )

        with self._lock:

            project = self.require_project(project_id)

            goal = ProjectGoal(
                goal_id=self._new_id("goal"),
                title=title,
                description=str(description).strip(),
                priority=priority,
                deadline=deadline,
                tags=self._clean_list(tags),
                metadata=dict(metadata or {}),
            )

            project.goals.append(goal)

            self._touch(project)

            self._record_event(
                project,
                "goal_added",
                f"Goal added: {title}",
                related_id=goal.goal_id,
            )

            return goal

    def update_goal(
        self,
        project_id: str,
        goal_id: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[ProjectItemStatus] = None,
        priority: Optional[ProjectPriority] = None,
        progress: Optional[float] = None,
        deadline: Optional[str] = None,
    ) -> ProjectGoal:
        """Update a project goal."""

        with self._lock:

            project = self.require_project(project_id)

            goal = self._find(
                project.goals,
                "goal_id",
                goal_id,
            )

            if goal is None:
                raise ProjectMemoryError(
                    f"Goal not found: {goal_id}"
                )

            if title is not None:
                goal.title = str(title).strip()

            if description is not None:
                goal.description = str(description).strip()

            if status is not None:
                goal.status = status

            if priority is not None:
                goal.priority = priority

            if progress is not None:
                goal.progress = self._clamp_percent(progress)

            if deadline is not None:
                goal.deadline = deadline

            goal.updated_at = self._utc_now()

            self._touch(project)

            return goal

    # ========================================================================
    # DECISIONS
    # ========================================================================

    def add_decision(
        self,
        project_id: str,
        title: str,
        decision: str,
        *,
        reasoning: str = "",
        alternatives: Optional[Iterable[str]] = None,
        impact: str = "",
        decided_by: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProjectDecision:
        """Record a project decision."""

        title = str(title).strip()
        decision = str(decision).strip()

        if not title:
            raise ProjectValidationError(
                "Decision title cannot be empty."
            )

        if not decision:
            raise ProjectValidationError(
                "Decision cannot be empty."
            )

        with self._lock:

            project = self.require_project(project_id)

            item = ProjectDecision(
                decision_id=self._new_id("decision"),
                title=title,
                decision=decision,
                reasoning=str(reasoning).strip(),
                alternatives=self._clean_list(alternatives),
                impact=str(impact).strip(),
                decided_by=decided_by,
                metadata=dict(metadata or {}),
            )

            project.decisions.append(item)

            self._touch(project)

            self._record_event(
                project,
                "decision_recorded",
                f"Decision recorded: {title}",
                related_id=item.decision_id,
            )

            return item

    # ========================================================================
    # MILESTONES
    # ========================================================================

    def add_milestone(
        self,
        project_id: str,
        title: str,
        description: str = "",
        *,
        deadline: Optional[str] = None,
        dependencies: Optional[Iterable[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProjectMilestone:
        """Add a project milestone."""

        title = str(title).strip()

        if not title:
            raise ProjectValidationError(
                "Milestone title cannot be empty."
            )

        with self._lock:

            project = self.require_project(project_id)

            milestone = ProjectMilestone(
                milestone_id=self._new_id("milestone"),
                title=title,
                description=str(description).strip(),
                deadline=deadline,
                dependencies=self._clean_list(dependencies),
                metadata=dict(metadata or {}),
            )

            project.milestones.append(milestone)

            self._touch(project)

            self._record_event(
                project,
                "milestone_added",
                f"Milestone added: {title}",
                related_id=milestone.milestone_id,
            )

            return milestone

    def update_milestone(
        self,
        project_id: str,
        milestone_id: str,
        *,
        status: Optional[ProjectItemStatus] = None,
        progress: Optional[float] = None,
        deadline: Optional[str] = None,
    ) -> ProjectMilestone:
        """Update milestone progress/status."""

        with self._lock:

            project = self.require_project(project_id)

            milestone = self._find(
                project.milestones,
                "milestone_id",
                milestone_id,
            )

            if milestone is None:
                raise ProjectMemoryError(
                    f"Milestone not found: {milestone_id}"
                )

            if status is not None:
                milestone.status = status

                if status == ProjectItemStatus.COMPLETED:
                    milestone.progress = 100.0
                    milestone.completed_at = self._utc_now()

            if progress is not None:
                milestone.progress = self._clamp_percent(progress)

                if milestone.progress >= 100.0:
                    milestone.status = ProjectItemStatus.COMPLETED
                    milestone.completed_at = self._utc_now()

            if deadline is not None:
                milestone.deadline = deadline

            milestone.updated_at = self._utc_now()

            self._touch(project)

            return milestone

    # ========================================================================
    # FILES
    # ========================================================================

    def add_file(
        self,
        project_id: str,
        path: str,
        *,
        description: str = "",
        file_type: Optional[str] = None,
        module: Optional[str] = None,
        role: Optional[str] = None,
        status: ProjectItemStatus = ProjectItemStatus.OPEN,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProjectFile:
        """Register a project file/module."""

        path = str(path).strip()

        if not path:
            raise ProjectValidationError(
                "File path cannot be empty."
            )

        with self._lock:

            project = self.require_project(project_id)

            existing = next(
                (
                    item
                    for item in project.files
                    if item.path == path
                ),
                None,
            )

            if existing is not None:
                existing.description = description
                existing.file_type = file_type
                existing.module = module
                existing.role = role
                existing.status = status
                existing.tags = self._clean_list(tags)
                existing.metadata.update(metadata or {})
                existing.updated_at = self._utc_now()

                self._touch(project)

                return existing

            item = ProjectFile(
                file_id=self._new_id("file"),
                path=path,
                description=str(description).strip(),
                file_type=file_type,
                module=module,
                role=role,
                status=status,
                tags=self._clean_list(tags),
                metadata=dict(metadata or {}),
            )

            project.files.append(item)

            self._touch(project)

            self._record_event(
                project,
                "file_added",
                f"Project file registered: {path}",
                related_id=item.file_id,
            )

            return item

    def remove_file(
        self,
        project_id: str,
        file_id: str,
    ) -> bool:
        """Remove a project file from memory."""

        with self._lock:

            project = self.require_project(project_id)

            before = len(project.files)

            project.files = [
                item
                for item in project.files
                if item.file_id != file_id
            ]

            if len(project.files) == before:
                return False

            self._touch(project)

            return True

    # ========================================================================
    # TASKS
    # ========================================================================

    def add_task(
        self,
        project_id: str,
        title: str,
        description: str = "",
        *,
        priority: ProjectPriority = ProjectPriority.NORMAL,
        assigned_agent: Optional[str] = None,
        dependencies: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProjectTask:
        """Add a task to a project."""

        title = str(title).strip()

        if not title:
            raise ProjectValidationError(
                "Task title cannot be empty."
            )

        with self._lock:

            project = self.require_project(project_id)

            task = ProjectTask(
                task_id=self._new_id("project_task"),
                title=title,
                description=str(description).strip(),
                priority=priority,
                assigned_agent=assigned_agent,
                dependencies=self._clean_list(dependencies),
                tags=self._clean_list(tags),
                metadata=dict(metadata or {}),
            )

            project.tasks.append(task)

            self._touch(project)

            self._record_event(
                project,
                "task_added",
                f"Project task added: {title}",
                related_id=task.task_id,
            )

            return task

    def update_task(
        self,
        project_id: str,
        task_id: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[ProjectItemStatus] = None,
        priority: Optional[ProjectPriority] = None,
        assigned_agent: Optional[str] = None,
    ) -> ProjectTask:
        """Update a project task."""

        with self._lock:

            project = self.require_project(project_id)

            task = self._find(
                project.tasks,
                "task_id",
                task_id,
            )

            if task is None:
                raise ProjectMemoryError(
                    f"Task not found: {task_id}"
                )

            if title is not None:
                task.title = str(title).strip()

            if description is not None:
                task.description = str(description).strip()

            if status is not None:
                task.status = status

                if status == ProjectItemStatus.COMPLETED:
                    task.completed_at = self._utc_now()

            if priority is not None:
                task.priority = priority

            if assigned_agent is not None:
                task.assigned_agent = assigned_agent

            task.updated_at = self._utc_now()

            self._touch(project)

            return task

    # ========================================================================
    # NOTES
    # ========================================================================

    def add_note(
        self,
        project_id: str,
        title: str,
        content: str,
        *,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProjectNote:
        """Store free-form project knowledge."""

        title = str(title).strip()
        content = str(content).strip()

        if not title:
            raise ProjectValidationError(
                "Note title cannot be empty."
            )

        if not content:
            raise ProjectValidationError(
                "Note content cannot be empty."
            )

        with self._lock:

            project = self.require_project(project_id)

            note = ProjectNote(
                note_id=self._new_id("note"),
                title=title,
                content=content,
                tags=self._clean_list(tags),
                metadata=dict(metadata or {}),
            )

            project.notes.append(note)

            self._touch(project)

            self._record_event(
                project,
                "note_added",
                f"Project note added: {title}",
                related_id=note.note_id,
            )

            return note

    # ========================================================================
    # REQUIREMENTS
    # ========================================================================

    def add_requirement(
        self,
        project_id: str,
        title: str,
        description: str = "",
        *,
        priority: ProjectPriority = ProjectPriority.NORMAL,
        source: Optional[str] = None,
        acceptance_criteria: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProjectRequirement:
        """Add a project requirement."""

        title = str(title).strip()

        if not title:
            raise ProjectValidationError(
                "Requirement title cannot be empty."
            )

        with self._lock:

            project = self.require_project(project_id)

            requirement = ProjectRequirement(
                requirement_id=self._new_id("requirement"),
                title=title,
                description=str(description).strip(),
                priority=priority,
                source=source,
                acceptance_criteria=self._clean_list(
                    acceptance_criteria
                ),
                tags=self._clean_list(tags),
                metadata=dict(metadata or {}),
            )

            project.requirements.append(requirement)

            self._touch(project)

            self._record_event(
                project,
                "requirement_added",
                f"Requirement added: {title}",
                related_id=requirement.requirement_id,
            )

            return requirement

    # ========================================================================
    # CONTEXT
    # ========================================================================

    def set_context(
        self,
        project_id: str,
        key: str,
        value: Any,
    ) -> None:
        """Store arbitrary structured project context."""

        key = str(key).strip()

        if not key:
            raise ProjectValidation


