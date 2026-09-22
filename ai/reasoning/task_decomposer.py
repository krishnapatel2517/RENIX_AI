"""
RENIX AI
Task Decomposer

Breaks complex user requests into smaller, ordered,
dependency-aware tasks that RENIX agents can execute.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


# ============================================================================
# ENUMS
# ============================================================================


class TaskStatus(str, Enum):
    """Status of a decomposed task."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Priority of a task."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# TASK
# ============================================================================


@dataclass
class DecomposedTask:
    """One executable task produced by the decomposer."""

    id: str

    title: str

    description: str = ""

    action: Optional[str] = None

    agent: Optional[str] = None

    tool: Optional[str] = None

    dependencies: List[str] = field(
        default_factory=list
    )

    priority: TaskPriority = TaskPriority.NORMAL

    status: TaskStatus = TaskStatus.PENDING

    optional: bool = False

    input_data: Dict[str, Any] = field(
        default_factory=dict
    )

    output_data: Dict[str, Any] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    error: Optional[str] = None

    # ========================================================================
    # STATE
    # ========================================================================

    def mark_ready(self) -> None:
        """Mark task as ready for execution."""

        if self.status == TaskStatus.PENDING:
            self.status = TaskStatus.READY

    def start(self) -> None:
        """Start task execution."""

        self.status = TaskStatus.RUNNING
        self.error = None

    def complete(
        self,
        output: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark task as completed."""

        self.status = TaskStatus.COMPLETED

        if output:
            self.output_data.update(output)

        self.error = None

    def fail(
        self,
        error: Exception | str,
    ) -> None:
        """Mark task as failed."""

        self.status = TaskStatus.FAILED
        self.error = str(error)

    def skip(
        self,
        reason: Optional[str] = None,
    ) -> None:
        """Skip task."""

        self.status = TaskStatus.SKIPPED

        if reason:
            self.metadata["skip_reason"] = reason

    def cancel(self) -> None:
        """Cancel task."""

        self.status = TaskStatus.CANCELLED

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task."""

        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "action": self.action,
            "agent": self.agent,
            "tool": self.tool,
            "dependencies": list(self.dependencies),
            "priority": self.priority.value,
            "status": self.status.value,
            "optional": self.optional,
            "input_data": dict(self.input_data),
            "output_data": dict(self.output_data),
            "metadata": dict(self.metadata),
            "error": self.error,
        }


# ============================================================================
# TASK GRAPH
# ============================================================================


@dataclass
class TaskGraph:
    """
    Dependency graph containing all decomposed tasks.
    """

    id: str

    goal: str

    tasks: List[DecomposedTask] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    # ========================================================================
    # TASK MANAGEMENT
    # ========================================================================

    def add_task(
        self,
        task: DecomposedTask,
    ) -> None:
        """Add a task to the graph."""

        if self.get_task(task.id) is not None:
            raise ValueError(
                f"Duplicate task ID: {task.id}"
            )

        self.tasks.append(task)

    def get_task(
        self,
        task_id: str,
    ) -> Optional[DecomposedTask]:
        """Get task by ID."""

        for task in self.tasks:
            if task.id == task_id:
                return task

        return None

    def require_task(
        self,
        task_id: str,
    ) -> DecomposedTask:
        """Get task or raise an error."""

        task = self.get_task(task_id)

        if task is None:
            raise KeyError(
                f"Unknown task: {task_id}"
            )

        return task

    def remove_task(
        self,
        task_id: str,
    ) -> bool:
        """Remove a task and its dependency references."""

        task = self.get_task(task_id)

        if task is None:
            return False

        self.tasks.remove(task)

        for other in self.tasks:
            if task_id in other.dependencies:
                other.dependencies.remove(task_id)

        return True

    # ========================================================================
    # DEPENDENCIES
    # ========================================================================

    def dependencies_completed(
        self,
        task: DecomposedTask,
    ) -> bool:
        """Check whether all task dependencies are completed."""

        for dependency_id in task.dependencies:

            dependency = self.get_task(
                dependency_id
            )

            if dependency is None:
                return False

            if dependency.status not in {
                TaskStatus.COMPLETED,
                TaskStatus.SKIPPED,
            }:
                return False

        return True

    def get_ready_tasks(
        self,
    ) -> List[DecomposedTask]:
        """Return tasks that are ready to execute."""

        ready: List[DecomposedTask] = []

        for task in self.tasks:

            if task.status not in {
                TaskStatus.PENDING,
                TaskStatus.READY,
            }:
                continue

            if self.dependencies_completed(task):
                task.mark_ready()
                ready.append(task)

        return sorted(
            ready,
            key=lambda task: self._priority_value(
                task.priority
            ),
            reverse=True,
        )

    @staticmethod
    def _priority_value(
        priority: TaskPriority,
    ) -> int:
        """Convert priority to a numeric value."""

        return {
            TaskPriority.LOW: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.HIGH: 3,
            TaskPriority.CRITICAL: 4,
        }.get(priority, 2)

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def validate(self) -> List[str]:
        """Validate the task dependency graph."""

        errors: List[str] = []

        task_ids = {
            task.id
            for task in self.tasks
        }

        for task in self.tasks:

            for dependency in task.dependencies:

                if dependency not in task_ids:
                    errors.append(
                        (
                            f"Task '{task.id}' depends "
                            f"on missing task '{dependency}'."
                        )
                    )

                if dependency == task.id:
                    errors.append(
                        (
                            f"Task '{task.id}' cannot "
                            "depend on itself."
                        )
                    )

        if self._has_cycle():
            errors.append(
                "Task dependency graph contains a cycle."
            )

        return errors

    def _has_cycle(self) -> bool:
        """Detect dependency cycles."""

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> bool:

            if task_id in visiting:
                return True

            if task_id in visited:
                return False

            visiting.add(task_id)

            task = self.get_task(task_id)

            if task:

                for dependency in task.dependencies:

                    if visit(dependency):
                        return True

            visiting.remove(task_id)
            visited.add(task_id)

            return False

        for task in self.tasks:

            if visit(task.id):
                return True

        return False

    # ========================================================================
    # PROGRESS
    # ========================================================================

    @property
    def total_tasks(self) -> int:
        """Total number of tasks."""

        return len(self.tasks)

    @property
    def completed_tasks(self) -> int:
        """Number of completed tasks."""

        return sum(
            task.status == TaskStatus.COMPLETED
            for task in self.tasks
        )

    @property
    def failed_tasks(self) -> int:
        """Number of failed tasks."""

        return sum(
            task.status == TaskStatus.FAILED
            for task in self.tasks
        )

    @property
    def progress(self) -> float:
        """Return progress from 0 to 1."""

        if not self.tasks:
            return 1.0

        return self.completed_tasks / len(self.tasks)

    @property
    def is_complete(self) -> bool:
        """Check whether every task is finished."""

        if not self.tasks:
            return True

        return all(
            task.status
            in {
                TaskStatus.COMPLETED,
                TaskStatus.SKIPPED,
            }
            for task in self.tasks
        )

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the complete graph."""

        return {
            "id": self.id,
            "goal": self.goal,
            "tasks": [
                task.to_dict()
                for task in self.tasks
            ],
            "metadata": dict(self.metadata),
            "progress": self.progress,
        }


# ============================================================================
# TASK DECOMPOSER
# ============================================================================


class TaskDecomposer:
    """
    RENIX task decomposition engine.

    Converts complex requests into smaller executable tasks.

    An optional callback can be connected to an LLM/planning engine
    to generate advanced decompositions.
    """

    def __init__(
        self,
        *,
        decomposition_callback: Optional[Any] = None,
    ) -> None:

        self.decomposition_callback = (
            decomposition_callback
        )

        self._graphs: Dict[str, TaskGraph] = {}

    # ========================================================================
    # MAIN DECOMPOSITION
    # ========================================================================

    def decompose(
        self,
        goal: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskGraph:
        """
        Decompose a goal into executable tasks.
        """

        if not goal or not goal.strip():
            raise ValueError(
                "Goal cannot be empty."
            )

        if self.decomposition_callback:

            generated = self.decomposition_callback(
                goal=goal,
                context=context or {},
            )

            graph = self._normalize_generated_graph(
                goal,
                generated,
            )

        else:
            graph = self._create_basic_graph(
                goal
            )

        errors = graph.validate()

        if errors:
            raise ValueError(
                "; ".join(errors)
            )

        self._graphs[graph.id] = graph

        return graph

    # ========================================================================
    # CALLBACK NORMALIZATION
    # ========================================================================

    def _normalize_generated_graph(
        self,
        goal: str,
        generated: Any,
    ) -> TaskGraph:
        """
        Convert callback output into a TaskGraph.
        """

        if isinstance(
            generated,
            TaskGraph,
        ):
            return generated

        graph = TaskGraph(
            id=self._generate_id("graph"),
            goal=goal.strip(),
        )

        if isinstance(
            generated,
            dict,
        ):
            raw_tasks = generated.get(
                "tasks",
                [],
            )

            graph.metadata.update(
                generated.get(
                    "metadata",
                    {},
                )
            )

        elif isinstance(
            generated,
            (list, tuple),
        ):
            raw_tasks = generated

        else:
            raise ValueError(
                (
                    "Decomposition callback must return "
                    "a TaskGraph, dictionary or sequence."
                )
            )

        for index, item in enumerate(
            raw_tasks,
            start=1,
        ):

            task = self._normalize_task(
                item,
                index,
            )

            graph.add_task(task)

        return graph

    def _normalize_task(
        self,
        item: Any,
        index: int,
    ) -> DecomposedTask:
        """Convert arbitrary task data into DecomposedTask."""

        if isinstance(
            item,
            DecomposedTask,
        ):
            return item

        if isinstance(
            item,
            str,
        ):
            return DecomposedTask(
                id=f"task_{index}",
                title=item,
                description=item,
            )

        if isinstance(
            item,
            dict,
        ):

            return DecomposedTask(
                id=item.get(
                    "id",
                    f"task_{index}",
                ),
                title=item.get(
                    "title",
                    f"Task {index}",
                ),
                description=item.get(
                    "description",
                    "",
                ),
                action=item.get(
                    "action"
                ),
                agent=item.get(
                    "agent"
                ),
                tool=item.get(
                    "tool"
                ),
                dependencies=list(
                    item.get(
                        "dependencies",
                        [],
                    )
                ),
                priority=self._parse_priority(
                    item.get(
                        "priority",
                        "normal",
                    )
                ),
                optional=bool(
                    item.get(
                        "optional",
                        False,
                    )
                ),
                input_data=dict(
                    item.get(
                        "input_data",
                        {},
                    )
                ),
                metadata=dict(
                    item.get(
                        "metadata",
                        {},
                    )
                ),
            )

        raise ValueError(
            (
                "Unsupported task returned by "
                f"decomposition callback at index {index}."
            )
        )

    @staticmethod
    def _parse_priority(
        value: Any,
    ) -> TaskPriority:
        """Convert a value into TaskPriority."""

        if isinstance(
            value,
            TaskPriority,
        ):
            return value

        try:
            return TaskPriority(
                str(value).strip().lower()
            )
        except ValueError:
            return TaskPriority.NORMAL

    # ========================================================================
    # BASIC FALLBACK
    # ========================================================================

    def _create_basic_graph(
        self,
        goal: str,
    ) -> TaskGraph:
        """
        Create a deterministic fallback decomposition.

        This keeps RENIX functional even before an external AI
        reasoning callback is connected.
        """

        graph = TaskGraph(
            id=self._generate_id("graph"),
            goal=goal.strip(),
        )

        understand = DecomposedTask(
            id="task_1",
            title="Understand the request",
            description=(
                "Identify the user's objective, "
                "requirements and constraints."
            ),
            action="understand_request",
            priority=TaskPriority.HIGH,
        )

        execute = DecomposedTask(
            id="task_2",
            title="Execute the request",
            description=(
                "Perform the actions required "
                "to satisfy the user's request."
            ),
            action="execute_request",
            dependencies=[
                "task_1"
            ],
            priority=TaskPriority.HIGH,
        )

        verify = DecomposedTask(
            id="task_3",
            title="Verify the result",
            description=(
                "Check that the requested result "
                "was successfully produced."
            ),
            action="verify_result",
            dependencies=[
                "task_2"
            ],
            priority=TaskPriority.NORMAL,
        )

        graph.add_task(understand)
        graph.add_task(execute)
        graph.add_task(verify)

        return graph

    # ========================================================================
    # GRAPH MANAGEMENT
    # ========================================================================

    def get_graph(
        self,
        graph_id: str,
    ) -> Optional[TaskGraph]:
        """Get a graph by ID."""

        return self._graphs.get(
            graph_id
        )

    def require_graph(
        self,
        graph_id: str,
    ) -> TaskGraph:
        """Get a graph or raise an error."""

        graph = self.get_graph(graph_id)

        if graph is None:
            raise KeyError(
                f"Unknown task graph: {graph_id}"
            )

        return graph

    def delete_graph(
        self,
        graph_id: str,
    ) -> bool:
        """Delete a graph."""

        return (
            self._graphs.pop(
                graph_id,
                None,
            )
            is not None
        )

    def list_graphs(
        self,
    ) -> List[TaskGraph]:
        """Return all task graphs."""

        return list(
            self._graphs.values()
        )

    # ========================================================================
    # TASK ACCESS
    # ========================================================================

    def get_next_tasks(
        self,
        graph_id: str,
    ) -> List[DecomposedTask]:
        """Get all currently executable tasks."""

        graph = self.require_graph(graph_id)

        return graph.get_ready_tasks()

    def get_next_task(
        self,
        graph_id: str,
    ) -> Optional[DecomposedTask]:
        """Get the highest-priority executable task."""

        tasks = self.get_next_tasks(graph_id)

        if not tasks:
            return None

        return tasks[0]

    # ========================================================================
    # TASK STATE
    # ========================================================================

    def start_task(
        self,
        graph_id: str,
        task_id: str,
    ) -> DecomposedTask:
        """Start a task."""

        graph = self.require_graph(graph_id)

        task = graph.require_task(task_id)

        if not graph.dependencies_completed(task):
            raise RuntimeError(
                (
                    f"Task '{task_id}' cannot start "
                    "because its dependencies are incomplete."
                )
            )

        task.start()

        return task

    def complete_task(
        self,
        graph_id: str,
        task_id: str,
        *,
        output: Optional[Dict[str, Any]] = None,
    ) -> TaskGraph:
        """Complete a task."""

        graph = self.require_graph(graph_id)

        task = graph.require_task(task_id)

        task.complete(output)

        return graph

    def fail_task(
        self,
        graph_id: str,
        task_id: str,
        error: Exception | str,
    ) -> TaskGraph:
        """Fail a task."""

        graph = self.require_graph(graph_id)

        task = graph.require_task(task_id)

        task.fail(error)

        return graph

    def skip_task(
        self,
        graph_id: str,
        task_id: str,
        reason: Optional[str] = None,
    ) -> TaskGraph:
        """Skip a task."""

        graph = self.require_graph(graph_id)

        task = graph.require_task(task_id)

        if not task.optional and reason is None:
            raise RuntimeError(
                (
                    f"Task '{task_id}' is not optional. "
                    "Provide a reason before skipping it."
                )
            )

        task.skip(reason)

        return graph

    def cancel_graph(
        self,
        graph_id: str,
    ) -> TaskGraph:
        """Cancel all unfinished tasks in a graph."""

        graph = self.require_graph(graph_id)

        for task in graph.tasks:

            if task.status in {
                TaskStatus.PENDING,
                TaskStatus.READY,
                TaskStatus.RUNNING,
            }:
                task.cancel()

        return graph

    # ========================================================================
    # DYNAMIC TASKS
    # ========================================================================

    def add_task(
        self,
        graph_id: str,
        title: str,
        *,
        description: str = "",
        action: Optional[str] = None,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
        dependencies: Optional[
            Sequence[str]
        ] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        optional: bool = False,
        input_data: Optional[
            Dict[str, Any]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> DecomposedTask:
        """
        Add a new task to an existing graph.
        """

        graph = self.require_graph(graph_id)

        task = DecomposedTask(
            id=self._generate_id("task"),
            title=title,
            description=description,
            action=action,
            agent=agent,
            tool=tool,
            dependencies=list(
                dependencies or []
            ),
            priority=priority,
            optional=optional,
            input_data=dict(
                input_data or {}
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        graph.add_task(task)

        errors = graph.validate()

        if errors:
            graph.remove_task(task.id)

            raise ValueError(
                "; ".join(errors)
            )

        return task

    # ========================================================================
    # ORDERING
    # ========================================================================

    def execution_order(
        self,
        graph_id: str,
    ) -> List[DecomposedTask]:
        """
        Return tasks in dependency-safe execution order.

        Uses a topological sort.
        """

        graph = self.require_graph(graph_id)

        errors = graph.validate()

        if errors:
            raise ValueError(
                "; ".join(errors)
            )

        ordered: List[
            DecomposedTask
        ] = []

        completed: set[str] = set()

        remaining = {
            task.id: task
            for task in graph.tasks
        }

        while remaining:

            progress = False

            candidates = sorted(
                remaining.values(),
                key=lambda task: graph._priority_value(
                    task.priority
                ),
                reverse=True,
            )

            for task in candidates:

                if all(
                    dependency in completed
                    for dependency
                    in task.dependencies
                ):
                    ordered.append(task)

                    completed.add(task.id)

                    remaining.pop(task.id)

                    progress = True

            if not progress:
                raise RuntimeError(
                    "Unable to determine a valid task execution order."
                )

        return ordered

    # ========================================================================
    # REPLANNING
    # ========================================================================

    def replan(
        self,
        graph_id: str,
        *,
        additional_goal: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskGraph:
        """
        Re-run decomposition using the existing graph as context.
        """

        graph = self.require_graph(graph_id)

        goal = graph.goal

        if additional_goal:
            goal = (
                f"{goal}\n\nAdditional requirement: "
                f"{additional_goal}"
            )

        merged_context = {
            "previous_graph": graph.to_dict(),
            **(context or {}),
        }

        return self.decompose(
            goal,
            context=merged_context,
        )

    # ========================================================================
    # IDENTIFIERS
    # ========================================================================

    @staticmethod
    def _generate_id(
        prefix: str,
    ) -> str:
        """Generate a unique identifier."""

        return (
            f"{prefix}_{uuid.uuid4().hex}"
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def decompose_task(
    goal: str,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> TaskGraph:
    """
    Convenience function for decomposing a goal.
    """

    decomposer = TaskDecomposer()

    return decomposer.decompose(
        goal,
        context=context,
    )


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================


__all__ = [
    "TaskStatus",
    "TaskPriority",
    "DecomposedTask",
    "TaskGraph",
    "TaskDecomposer",
    "decompose_task",
]


