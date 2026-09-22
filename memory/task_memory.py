"""
RENIX AI - Task Memory

Stores, retrieves, updates, and manages memories associated with tasks.

Responsibilities:
- Persist task-related memory.
- Track task lifecycle and execution history.
- Store task context, inputs, outputs, errors, and results.
- Link memories to projects and conversations.
- Support task retrieval and semantic-friendly metadata.
- Provide safe cleanup/export functionality.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


logger = logging.getLogger("RENIX.memory.task_memory")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _safe_json(value: Any) -> Any:
    """
    Convert common Python objects into JSON-safe values.

    This prevents task-memory persistence from failing because a caller
    supplied an object that is not directly serializable.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): _safe_json(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_safe_json(item) for item in value]

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _safe_json(value.to_dict())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return _safe_json(vars(value))
        except Exception:
            pass

    return str(value)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TaskMemoryRecord:
    """
    A persistent memory record belonging to a task.
    """

    memory_id: str
    task_id: str

    title: str = ""
    description: str = ""

    status: str = "created"

    task_type: str = ""
    priority: str = "normal"

    project_id: Optional[str] = None
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None

    user_input: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    actions: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)

    result: Any = None
    error: Optional[str] = None

    tags: List[str] = field(default_factory=list)

    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    completed_at: Optional[str] = None

    success: Optional[bool] = None

    importance: float = 0.5
    archived: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return _safe_json(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskMemoryRecord":
        """Create a record from persisted dictionary data."""
        allowed_fields = {
            field_name
            for field_name in cls.__dataclass_fields__
        }

        cleaned = {
            key: value
            for key, value in data.items()
            if key in allowed_fields
        }

        return cls(**cleaned)


# ---------------------------------------------------------------------------
# Task Memory Manager
# ---------------------------------------------------------------------------

class TaskMemory:
    """
    Task-specific memory manager for RENIX.

    The class intentionally keeps its persistence layer simple and
    dependency-free. Task memories are stored as JSON records under:

        RENIX/data/memory/tasks.json

    The manager can later be connected to the broader RENIX memory system,
    vector store, database, or semantic retrieval layer.
    """

    DEFAULT_FILENAME = "tasks.json"

    def __init__(
        self,
        storage_path: Optional[str | Path] = None,
        max_records: int = 10000,
        auto_save: bool = True,
    ) -> None:
        self.max_records = max(1, int(max_records))
        self.auto_save = bool(auto_save)

        if storage_path is None:
            storage_path = (
                Path(__file__).resolve().parents[1]
                / "data"
                / "memory"
                / self.DEFAULT_FILENAME
            )

        self.storage_path = Path(storage_path)

        self._records: Dict[str, TaskMemoryRecord] = {}
        self._lock = threading.RLock()

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ensure_storage(self) -> None:
        """Create the parent directory and storage file when required."""
        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.storage_path.exists():
            self.storage_path.write_text(
                "{}",
                encoding="utf-8",
            )

    def _load(self) -> None:
        """Load task memories from disk."""
        with self._lock:
            try:
                self._ensure_storage()

                raw = self.storage_path.read_text(
                    encoding="utf-8"
                ).strip()

                if not raw:
                    self._records = {}
                    return

                data = json.loads(raw)

                if not isinstance(data, dict):
                    logger.warning(
                        "Task memory storage is not a dictionary. "
                        "Starting with empty task memory."
                    )
                    self._records = {}
                    return

                loaded: Dict[str, TaskMemoryRecord] = {}

                for memory_id, record_data in data.items():
                    try:
                        if not isinstance(record_data, dict):
                            continue

                        record = TaskMemoryRecord.from_dict(record_data)

                        if not record.memory_id:
                            record.memory_id = str(memory_id)

                        loaded[record.memory_id] = record

                    except Exception as exc:
                        logger.warning(
                            "Failed to load task memory '%s': %s",
                            memory_id,
                            exc,
                        )

                self._records = loaded

            except json.JSONDecodeError as exc:
                logger.error(
                    "Invalid task memory JSON at %s: %s",
                    self.storage_path,
                    exc,
                )
                self._records = {}

            except Exception as exc:
                logger.exception(
                    "Unable to load task memory: %s",
                    exc,
                )
                self._records = {}

    def save(self) -> bool:
        """Persist all task memories to disk."""
        with self._lock:
            try:
                self._ensure_storage()

                payload = {
                    memory_id: record.to_dict()
                    for memory_id, record in self._records.items()
                }

                temporary_path = self.storage_path.with_suffix(
                    self.storage_path.suffix + ".tmp"
                )

                temporary_path.write_text(
                    json.dumps(
                        payload,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                temporary_path.replace(self.storage_path)

                return True

            except Exception as exc:
                logger.exception(
                    "Failed to save task memory: %s",
                    exc,
                )
                return False

    def _save_if_enabled(self) -> None:
        if self.auto_save:
            self.save()

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def create_task(
        self,
        task_id: Optional[str] = None,
        title: str = "",
        description: str = "",
        task_type: str = "",
        priority: str = "normal",
        user_input: str = "",
        project_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
        importance: float = 0.5,
    ) -> TaskMemoryRecord:
        """
        Create a new task memory record.
        """
        with self._lock:
            task_id = task_id or str(uuid.uuid4())

            memory_id = str(uuid.uuid4())
            now = _utc_now()

            record = TaskMemoryRecord(
                memory_id=memory_id,
                task_id=task_id,
                title=title,
                description=description,
                status="created",
                task_type=task_type,
                priority=priority,
                project_id=project_id,
                conversation_id=conversation_id,
                session_id=session_id,
                user_input=user_input,
                context=_safe_json(context or {}),
                metadata=_safe_json(metadata or {}),
                tags=list(tags or []),
                created_at=now,
                updated_at=now,
                importance=max(
                    0.0,
                    min(1.0, float(importance)),
                ),
            )

            self._records[memory_id] = record

            self._enforce_limit()
            self._save_if_enabled()

            return record

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(
        self,
        memory_id: str,
    ) -> Optional[TaskMemoryRecord]:
        """Retrieve a memory by memory ID."""
        with self._lock:
            return self._records.get(memory_id)

    def get_task(
        self,
        task_id: str,
    ) -> List[TaskMemoryRecord]:
        """Return all memory records associated with a task."""
        with self._lock:
            records = [
                record
                for record in self._records.values()
                if record.task_id == task_id
            ]

            return sorted(
                records,
                key=lambda item: item.created_at,
            )

    def latest_for_task(
        self,
        task_id: str,
    ) -> Optional[TaskMemoryRecord]:
        """Return the newest memory record for a task."""
        records = self.get_task(task_id)

        if not records:
            return None

        return records[-1]

    def all(
        self,
        include_archived: bool = False,
    ) -> List[TaskMemoryRecord]:
        """Return all task memories."""
        with self._lock:
            records = list(self._records.values())

            if not include_archived:
                records = [
                    record
                    for record in records
                    if not record.archived
                ]

            return sorted(
                records,
                key=lambda item: item.updated_at,
                reverse=True,
            )

    # ------------------------------------------------------------------
    # Updating
    # ------------------------------------------------------------------

    def update(
        self,
        memory_id: str,
        **updates: Any,
    ) -> Optional[TaskMemoryRecord]:
        """
        Update fields on an existing memory record.

        Unknown fields are placed into metadata rather than silently
        modifying the dataclass structure.
        """
        with self._lock:
            record = self._records.get(memory_id)

            if record is None:
                return None

            allowed_fields = set(
                TaskMemoryRecord.__dataclass_fields__.keys()
            )

            for key, value in updates.items():
                if key == "memory_id":
                    continue

                if key in allowed_fields:
                    setattr(
                        record,
                        key,
                        _safe_json(value),
                    )
                else:
                    record.metadata[key] = _safe_json(value)

            record.updated_at = _utc_now()

            self._save_if_enabled()

            return record

    def update_task(
        self,
        task_id: str,
        **updates: Any,
    ) -> List[TaskMemoryRecord]:
        """Update every memory record belonging to a task."""
        with self._lock:
            records = self.get_task(task_id)

            for record in records:
                self.update(
                    record.memory_id,
                    **updates,
                )

            return records

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def set_status(
        self,
        task_id: str,
        status: str,
        success: Optional[bool] = None,
    ) -> List[TaskMemoryRecord]:
        """
        Update task status.

        Terminal statuses automatically receive a completion timestamp.
        """
        terminal_statuses = {
            "completed",
            "failed",
            "cancelled",
            "canceled",
            "aborted",
        }

        with self._lock:
            records = self.get_task(task_id)

            for record in records:
                record.status = status
                record.updated_at = _utc_now()

                if success is not None:
                    record.success = bool(success)

                if status.lower() in terminal_statuses:
                    record.completed_at = _utc_now()

            self._save_if_enabled()

            return records

    def complete_task(
        self,
        task_id: str,
        result: Any = None,
    ) -> List[TaskMemoryRecord]:
        """Mark a task as successfully completed."""
        with self._lock:
            records = self.get_task(task_id)

            for record in records:
                record.status = "completed"
                record.success = True
                record.result = _safe_json(result)
                record.completed_at = _utc_now()
                record.updated_at = _utc_now()

            self._save_if_enabled()

            return records

    def fail_task(
        self,
        task_id: str,
        error: Any,
    ) -> List[TaskMemoryRecord]:
        """Mark a task as failed and store the error."""
        with self._lock:
            records = self.get_task(task_id)

            for record in records:
                record.status = "failed"
                record.success = False
                record.error = str(error)
                record.completed_at = _utc_now()
                record.updated_at = _utc_now()

            self._save_if_enabled()

            return records

    # ------------------------------------------------------------------
    # Actions and outputs
    # ------------------------------------------------------------------

    def add_action(
        self,
        task_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        result: Any = None,
        success: Optional[bool] = None,
    ) -> Optional[TaskMemoryRecord]:
        """
        Record an action performed while executing a task.
        """
        with self._lock:
            record = self.latest_for_task(task_id)

            if record is None:
                return None

            action_record = {
                "action_id": str(uuid.uuid4()),
                "action": action,
                "details": _safe_json(details or {}),
                "result": _safe_json(result),
                "success": success,
                "timestamp": _utc_now(),
            }

            record.actions.append(action_record)
            record.updated_at = _utc_now()

            self._save_if_enabled()

            return record

    def add_output(
        self,
        task_id: str,
        output: Any,
        output_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TaskMemoryRecord]:
        """
        Store an output generated by task execution.
        """
        with self._lock:
            record = self.latest_for_task(task_id)

            if record is None:
                return None

            output_record = {
                "output_id": str(uuid.uuid4()),
                "type": output_type,
                "content": _safe_json(output),
                "metadata": _safe_json(metadata or {}),
                "timestamp": _utc_now(),
            }

            record.outputs.append(output_record)
            record.updated_at = _utc_now()

            self._save_if_enabled()

            return record

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def set_context(
        self,
        task_id: str,
        context: Dict[str, Any],
        merge: bool = True,
    ) -> Optional[TaskMemoryRecord]:
        """Store execution context for a task."""
        with self._lock:
            record = self.latest_for_task(task_id)

            if record is None:
                return None

            safe_context = _safe_json(context)

            if merge:
                if isinstance(safe_context, dict):
                    record.context.update(safe_context)
            else:
                record.context = (
                    safe_context
                    if isinstance(safe_context, dict)
                    else {}
                )

            record.updated_at = _utc_now()

            self._save_if_enabled()

            return record

    def get_context(
        self,
        task_id: str,
    ) -> Dict[str, Any]:
        """Return the latest context associated with a task."""
        record = self.latest_for_task(task_id)

        if record is None:
            return {}

        return dict(record.context)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 20,
        include_archived: bool = False,
    ) -> List[TaskMemoryRecord]:
        """
        Perform lightweight keyword-based task-memory search.

        The dedicated semantic/vector retrieval layer can build on top of
        this method later.
        """
        if not query:
            return []

        query_terms = [
            term.lower()
            for term in query.split()
            if term.strip()
        ]

        if not query_terms:
            return []

        results = []

        with self._lock:
            for record in self._records.values():
                if record.archived and not include_archived:
                    continue

                searchable = " ".join(
                    [
                        record.task_id,
                        record.title,
                        record.description,
                        record.task_type,
                        record.user_input,
                        record.status,
                        " ".join(record.tags),
                        json.dumps(
                            record.context,
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            record.metadata,
                            ensure_ascii=False,
                        ),
                    ]
                ).lower()

                score = sum(
                    1
                    for term in query_terms
                    if term in searchable
                )

                if score:
                    results.append(
                        (
                            score,
                            record.updated_at,
                            record,
                        )
                    )

        results.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        return [
            item[2]
            for item in results[: max(1, limit)]
        ]

    # ------------------------------------------------------------------
    # Project / conversation retrieval
    # ------------------------------------------------------------------

    def get_by_project(
        self,
        project_id: str,
        include_archived: bool = False,
    ) -> List[TaskMemoryRecord]:
        """Retrieve memories belonging to a project."""
        with self._lock:
            records = [
                record
                for record in self._records.values()
                if record.project_id == project_id
                and (
                    include_archived
                    or not record.archived
                )
            ]

            return sorted(
                records,
                key=lambda item: item.updated_at,
                reverse=True,
            )

    def get_by_conversation(
        self,
        conversation_id: str,
        include_archived: bool = False,
    ) -> List[TaskMemoryRecord]:
        """Retrieve memories belonging to a conversation."""
        with self._lock:
            records = [
                record
                for record in self._records.values()
                if record.conversation_id == conversation_id
                and (
                    include_archived
                    or not record.archived
                )
            ]

            return sorted(
                records,
                key=lambda item: item.updated_at,
                reverse=True,
            )

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def add_tag(
        self,
        task_id: str,
        tag: str,
    ) -> Optional[TaskMemoryRecord]:
        """Add a tag to the latest task memory."""
        tag = str(tag).strip()

        if not tag:
            return self.latest_for_task(task_id)

        with self._lock:
            record = self.latest_for_task(task_id)

            if record is None:
                return None

            if tag not in record.tags:
                record.tags.append(tag)
                record.updated_at = _utc_now()

            self._save_if_enabled()

            return record

    def remove_tag(
        self,
        task_id: str,
        tag: str,
    ) -> Optional[TaskMemoryRecord]:
        """Remove a tag from the latest task memory."""
        with self._lock:
            record = self.latest_for_task(task_id)

            if record is None:
                return None

            if tag in record.tags:
                record.tags.remove(tag)
                record.updated_at = _utc_now()

            self._save_if_enabled()

            return record

    # ------------------------------------------------------------------
    # Archiving / deletion
    # ------------------------------------------------------------------

    def archive(
        self,
        memory_id: str,
    ) -> bool:
        """Archive a specific memory record."""
        with self._lock:
            record = self._records.get(memory_id)

            if record is None:
                return False

            record.archived = True
            record.updated_at = _utc_now()

            self._save_if_enabled()

            return True

    def archive_task(
        self,
        task_id: str,
    ) -> int:
        """Archive every memory associated with a task."""
        count = 0

        with self._lock:
            for record in self._records.values():
                if record.task_id == task_id:
                    record.archived = True
                    record.updated_at = _utc_now()
                    count += 1

            self._save_if_enabled()

        return count

    def delete(
        self,
        memory_id: str,
    ) -> bool:
        """Permanently delete one memory record."""
        with self._lock:
            if memory_id not in self._records:
                return False

            del self._records[memory_id]

            self._save_if_enabled()

            return True

    def delete_task(
        self,
        task_id: str,
    ) -> int:
        """Permanently delete all memory records belonging to a task."""
        with self._lock:
            memory_ids = [
                memory_id
                for memory_id, record in self._records.items()
                if record.task_id == task_id
            ]

            for memory_id in memory_ids:
                del self._records[memory_id]

            if memory_ids:
                self._save_if_enabled()

            return len(memory_ids)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def _enforce_limit(self) -> None:
        """Keep the storage bounded by removing oldest archived records."""
        if len(self._records) <= self.max_records:
            return

        records = sorted(
            self._records.values(),
            key=lambda record: (
                record.archived,
                record.updated_at,
            ),
        )

        excess = len(self._records) - self.max_records

        for record in records[:excess]:
            self._records.pop(
                record.memory_id,
                None,
            )

    def cleanup(
        self,
        archive_only: bool = True,
    ) -> int:
        """
        Remove old records when storage exceeds max_records.

        If archive_only=True, only archived records are eligible.
        """
        with self._lock:
            if len(self._records) <= self.max_records:
                return 0

            if archive_only:
                candidates = [
                    record
                    for record in self._records.values()
                    if record.archived
                ]
            else:
                candidates = list(self._records.values())

            candidates.sort(
                key=lambda record: record.updated_at
            )

            amount = min(
                len(candidates),
                len(self._records) - self.max_records,
            )

            for record in candidates[:amount]:
                self._records.pop(
                    record.memory_id,
                    None,
                )

            if amount:
                self._save_if_enabled()

            return amount

    def clear(
        self,
        include_archived: bool = True,
    ) -> int:
        """
        Clear task memory.

        By default this permanently clears all records.
        """
        with self._lock:
            if include_archived:
                count = len(self._records)
                self._records.clear()
            else:
                memory_ids = [
                    memory_id
                    for memory_id, record in self._records.items()
                    if not record.archived
                ]

                for memory_id in memory_ids:
                    del self._records[memory_id]

                count = len(memory_ids)

            self._save_if_enabled()

            return count

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        """Return task-memory statistics."""
        with self._lock:
            records = list(self._records.values())

            status_counts: Dict[str, int] = {}

            for record in records:
                status_counts[record.status] = (
                    status_counts.get(record.status, 0) + 1
                )

            successful = sum(
                1
                for record in records
                if record.success is True
            )

            failed = sum(
                1
                for record in records
                if record.success is False
            )

            return {
                "total_records": len(records),
                "active_records": sum(
                    1
                    for record in records
                    if not record.archived
                ),
                "archived_records": sum(
                    1
                    for record in records
                    if record.archived
                ),
                "successful_records": successful,
                "failed_records": failed,
                "status_counts": status_counts,
                "storage_path": str(self.storage_path),
                "max_records": self.max_records,
            }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_task(
        self,
        task_id: str,
    ) -> Dict[str, Any]:
        """
        Return a complete serializable representation of a task's memory.
        """
        records = self.get_task(task_id)

        return {
            "task_id": task_id,
            "record_count": len(records),
            "records": [
                record.to_dict()
                for record in records
            ],
            "exported_at": _utc_now(),
        }


# ---------------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------------

TaskMemoryManager = TaskMemory


__all__ = [
    "TaskMemory",
    "TaskMemoryManager",
    "TaskMemoryRecord",
]


