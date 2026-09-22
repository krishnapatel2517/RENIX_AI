"""
RENIX AI - Vector Store

Persistent vector storage abstraction for RENIX memory.

Responsibilities:
- Store embeddings with memory metadata.
- Add, update, delete and retrieve vectors.
- Perform cosine-similarity search.
- Support filtering by memory type, task, project, conversation, etc.
- Persist data locally without requiring an external vector database.
- Provide a clean interface for future FAISS/Chroma/Qdrant integration.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


logger = logging.getLogger("RENIX.memory.vector_store")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(k): _safe_json(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_safe_json(v) for v in value]

    return str(value)


@dataclass
class VectorRecord:
    """A single vector stored in RENIX memory."""

    vector_id: str
    embedding: List[float]

    text: str = ""
    memory_type: str = "semantic"

    metadata: Dict[str, Any] = field(default_factory=dict)

    memory_id: Optional[str] = None
    task_id: Optional[str] = None
    project_id: Optional[str] = None
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None

    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    importance: float = 0.5
    archived: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return _safe_json(
            {
                "vector_id": self.vector_id,
                "embedding": self.embedding,
                "text": self.text,
                "memory_type": self.memory_type,
                "metadata": self.metadata,
                "memory_id": self.memory_id,
                "task_id": self.task_id,
                "project_id": self.project_id,
                "conversation_id": self.conversation_id,
                "session_id": self.session_id,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "importance": self.importance,
                "archived": self.archived,
            }
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorRecord":
        return cls(
            vector_id=str(
                data.get("vector_id", uuid.uuid4())
            ),
            embedding=[
                float(x)
                for x in data.get("embedding", [])
            ],
            text=str(data.get("text", "")),
            memory_type=str(
                data.get("memory_type", "semantic")
            ),
            metadata=dict(
                data.get("metadata", {})
            ),
            memory_id=data.get("memory_id"),
            task_id=data.get("task_id"),
            project_id=data.get("project_id"),
            conversation_id=data.get("conversation_id"),
            session_id=data.get("session_id"),
            created_at=str(
                data.get("created_at", _utc_now())
            ),
            updated_at=str(
                data.get("updated_at", _utc_now())
            ),
            importance=float(
                data.get("importance", 0.5)
            ),
            archived=bool(
                data.get("archived", False)
            ),
        )


class VectorStore:
    """
    Lightweight persistent vector store.

    This implementation intentionally uses only Python's standard library.
    It can later be replaced by FAISS, Chroma, Qdrant or another vector
    database without changing the higher-level RENIX memory APIs.
    """

    DEFAULT_FILENAME = "vectors.json"

    def __init__(
        self,
        storage_path: Optional[str | Path] = None,
        dimension: Optional[int] = None,
        max_records: int = 50000,
        auto_save: bool = True,
    ) -> None:
        if storage_path is None:
            storage_path = (
                Path(__file__).resolve().parents[1]
                / "data"
                / "embeddings"
                / self.DEFAULT_FILENAME
            )

        self.storage_path = Path(storage_path)
        self.dimension = dimension
        self.max_records = max(1, int(max_records))
        self.auto_save = bool(auto_save)

        self._records: Dict[str, VectorRecord] = {}
        self._lock = threading.RLock()

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ensure_storage(self) -> None:
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
                        "Vector store is not a dictionary."
                    )
                    self._records = {}
                    return

                records: Dict[str, VectorRecord] = {}

                for vector_id, value in data.items():
                    try:
                        if not isinstance(value, dict):
                            continue

                        record = VectorRecord.from_dict(value)

                        if not record.vector_id:
                            record.vector_id = str(vector_id)

                        if record.embedding:
                            if self.dimension is None:
                                self.dimension = len(
                                    record.embedding
                                )

                        records[
                            record.vector_id
                        ] = record

                    except Exception as exc:
                        logger.warning(
                            "Could not load vector '%s': %s",
                            vector_id,
                            exc,
                        )

                self._records = records

            except json.JSONDecodeError:
                logger.error(
                    "Vector store contains invalid JSON."
                )
                self._records = {}

            except Exception as exc:
                logger.exception(
                    "Failed to load vector store: %s",
                    exc,
                )
                self._records = {}

    def save(self) -> bool:
        with self._lock:
            try:
                self._ensure_storage()

                payload = {
                    vector_id: record.to_dict()
                    for vector_id, record
                    in self._records.items()
                }

                temporary = self.storage_path.with_suffix(
                    self.storage_path.suffix + ".tmp"
                )

                temporary.write_text(
                    json.dumps(
                        payload,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                temporary.replace(
                    self.storage_path
                )

                return True

            except Exception as exc:
                logger.exception(
                    "Failed to save vector store: %s",
                    exc,
                )
                return False

    def _save_if_enabled(self) -> None:
        if self.auto_save:
            self.save()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _normalize_embedding(
        self,
        embedding: Iterable[float],
    ) -> List[float]:
        vector = [
            float(value)
            for value in embedding
        ]

        if not vector:
            raise ValueError(
                "Embedding cannot be empty."
            )

        if self.dimension is None:
            self.dimension = len(vector)

        if len(vector) != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {self.dimension}, "
                f"received {len(vector)}."
            )

        if not all(
            math.isfinite(value)
            for value in vector
        ):
            raise ValueError(
                "Embedding contains non-finite values."
            )

        return vector

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(
        a: Iterable[float],
        b: Iterable[float],
    ) -> float:
        """
        Calculate cosine similarity between two vectors.

        Returns:
            Value between -1 and 1.
        """
        a_list = list(a)
        b_list = list(b)

        if len(a_list) != len(b_list):
            raise ValueError(
                "Vectors must have equal dimensions."
            )

        dot = sum(
            x * y
            for x, y in zip(a_list, b_list)
        )

        norm_a = math.sqrt(
            sum(x * x for x in a_list)
        )

        norm_b = math.sqrt(
            sum(y * y for y in b_list)
        )

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(
        self,
        embedding: Iterable[float],
        text: str = "",
        memory_type: str = "semantic",
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
        task_id: Optional[str] = None,
        project_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        importance: float = 0.5,
        vector_id: Optional[str] = None,
    ) -> VectorRecord:
        """Add a vector to the store."""

        with self._lock:
            vector = self._normalize_embedding(
                embedding
            )

            record = VectorRecord(
                vector_id=vector_id or str(
                    uuid.uuid4()
                ),
                embedding=vector,
                text=text,
                memory_type=memory_type,
                metadata=_safe_json(
                    metadata or {}
                ),
                memory_id=memory_id,
                task_id=task_id,
                project_id=project_id,
                conversation_id=conversation_id,
                session_id=session_id,
                importance=max(
                    0.0,
                    min(1.0, float(importance)),
                ),
            )

            self._records[
                record.vector_id
            ] = record

            self._enforce_limit()
            self._save_if_enabled()

            return record

    # Compatibility alias.
    add_vector = add

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------

    def get(
        self,
        vector_id: str,
    ) -> Optional[VectorRecord]:
        with self._lock:
            return self._records.get(vector_id)

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def all(
        self,
        include_archived: bool = False,
    ) -> List[VectorRecord]:
        with self._lock:
            records = list(
                self._records.values()
            )

            if not include_archived:
                records = [
                    record
                    for record in records
                    if not record.archived
                ]

            return records

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        vector_id: str,
        embedding: Optional[Iterable[float]] = None,
        text: Optional[str] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
        archived: Optional[bool] = None,
        **extra_metadata: Any,
    ) -> Optional[VectorRecord]:
        with self._lock:
            record = self._records.get(
                vector_id
            )

            if record is None:
                return None

            if embedding is not None:
                record.embedding = (
                    self._normalize_embedding(
                        embedding
                    )
                )

            if text is not None:
                record.text = str(text)

            if memory_type is not None:
                record.memory_type = str(
                    memory_type
                )

            if metadata is not None:
                record.metadata = _safe_json(
                    metadata
                )

            if importance is not None:
                record.importance = max(
                    0.0,
                    min(1.0, float(importance)),
                )

            if archived is not None:
                record.archived = bool(
                    archived
                )

            if extra_metadata:
                record.metadata.update(
                    _safe_json(
                        extra_metadata
                    )
                )

            record.updated_at = _utc_now()

            self._save_if_enabled()

            return record

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: Iterable[float],
        top_k: int = 5,
        min_score: float = -1.0,
        filters: Optional[Dict[str, Any]] = None,
        include_archived: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search the vector store using cosine similarity.

        Returns dictionaries containing:

            vector_id
            score
            record
        """

        query = self._normalize_embedding(
            query_embedding
        )

        top_k = max(1, int(top_k))
        filters = filters or {}

        matches: List[Dict[str, Any]] = []

        with self._lock:
            for record in self._records.values():

                if (
                    record.archived
                    and not include_archived
                ):
                    continue

                if not self._matches_filters(
                    record,
                    filters,
                ):
                    continue

                score = self.cosine_similarity(
                    query,
                    record.embedding,
                )

                if score < min_score:
                    continue

                matches.append(
                    {
                        "vector_id": record.vector_id,
                        "score": score,
                        "record": record,
                    }
                )

        matches.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return matches[:top_k]

    # Compatibility aliases.
    similarity_search = search
    query = search

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _matches_filters(
        self,
        record: VectorRecord,
        filters: Dict[str, Any],
    ) -> bool:
        """
        Match standard RENIX memory filters.

        Supported fields:
        - memory_type
        - memory_id
        - task_id
        - project_id
        - conversation_id
        - session_id
        - archived
        - tags
        - arbitrary metadata keys
        """

        for key, expected in filters.items():

            if key == "memory_type":
                actual = record.memory_type

            elif key == "memory_id":
                actual = record.memory_id

            elif key == "task_id":
                actual = record.task_id

            elif key == "project_id":
                actual = record.project_id

            elif key == "conversation_id":
                actual = record.conversation_id

            elif key == "session_id":
                actual = record.session_id

            elif key == "archived":
                actual = record.archived

            elif key == "tags":
                actual = record.metadata.get(
                    "tags",
                    [],
                )

            else:
                actual = record.metadata.get(
                    key
                )

            if isinstance(expected, (list, tuple, set)):
                if isinstance(actual, (list, tuple, set)):
                    if not any(
                        item in actual
                        for item in expected
                    ):
                        return False
                elif actual not in expected:
                    return False
            else:
                if actual != expected:
                    return False

        return True

    # ------------------------------------------------------------------
    # Metadata-based retrieval
    # ------------------------------------------------------------------

    def get_by_memory_id(
        self,
        memory_id: str,
    ) -> List[VectorRecord]:
        return [
            record
            for record in self.all(
                include_archived=True
            )
            if record.memory_id == memory_id
        ]

    def get_by_task_id(
        self,
        task_id: str,
    ) -> List[VectorRecord]:
        return [
            record
            for record in self.all(
                include_archived=True
            )
            if record.task_id == task_id
        ]

    def get_by_project_id(
        self,
        project_id: str,
    ) -> List[VectorRecord]:
        return [
            record
            for record in self.all(
                include_archived=True
            )
            if record.project_id == project_id
        ]

    def get_by_conversation_id(
        self,
        conversation_id: str,
    ) -> List[VectorRecord]:
        return [
            record
            for record in self.all(
                include_archived=True
            )
            if record.conversation_id
            == conversation_id
        ]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(
        self,
        vector_id: str,
    ) -> bool:
        with self._lock:
            if vector_id not in self._records:
                return False

            del self._records[
                vector_id
            ]

            self._save_if_enabled()

            return True

    def delete_by_memory_id(
        self,
        memory_id: str,
    ) -> int:
        with self._lock:
            ids = [
                vector_id
                for vector_id, record
                in self._records.items()
                if record.memory_id == memory_id
            ]

            for vector_id in ids:
                del self._records[
                    vector_id
                ]

            if ids:
                self._save_if_enabled()

            return len(ids)

    def delete_by_task_id(
        self,
        task_id: str,
    ) -> int:
        with self._lock:
            ids = [
                vector_id
                for vector_id, record
                in self._records.items()
                if record.task_id == task_id
            ]

            for vector_id in ids:
                del self._records[
                    vector_id
                ]

            if ids:
                self._save_if_enabled()

            return len(ids)

    # ------------------------------------------------------------------
    # Archive
    # ------------------------------------------------------------------

    def archive(
        self,
        vector_id: str,
    ) -> bool:
        with self._lock:
            record = self._records.get(
                vector_id
            )

            if record is None:
                return False

            record.archived = True
            record.updated_at = _utc_now()

            self._save_if_enabled()

            return True

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def _enforce_limit(self) -> None:
        if len(self._records) <= self.max_records:
            return

        records = sorted(
            self._records.values(),
            key=lambda item: (
                not item.archived,
                item.updated_at,
            ),
        )

        excess = (
            len(self._records)
            - self.max_records
        )

        for record in records[:excess]:
            self._records.pop(
                record.vector_id,
                None,
            )

    def cleanup(
        self,
        archived_only: bool = True,
    ) -> int:
        with self._lock:
            if len(self._records) <= self.max_records:
                return 0

            if archived_only:
                candidates = [
                    record
                    for record in self._records.values()
                    if record.archived
                ]
            else:
                candidates = list(
                    self._records.values()
                )

            candidates.sort(
                key=lambda item:
                item.updated_at
            )

            amount = min(
                len(candidates),
                len(self._records)
                - self.max_records,
            )

            for record in candidates[:amount]:
                self._records.pop(
                    record.vector_id,
                    None,
                )

            if amount:
                self._save_if_enabled()

            return amount

    def clear(
        self,
        include_archived: bool = True,
    ) -> int:
        with self._lock:
            if include_archived:
                count = len(self._records)
                self._records.clear()

            else:
                ids = [
                    vector_id
                    for vector_id, record
                    in self._records.items()
                    if not record.archived
                ]

                for vector_id in ids:
                    del self._records[
                        vector_id
                    ]

                count = len(ids)

            self._save_if_enabled()

            return count

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def export_data(
        self,
        include_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """
        Export the vector store.

        Embeddings can be omitted when exporting metadata only.
        """
        with self._lock:
            records = []

            for record in self._records.values():
                data = record.to_dict()

                if not include_embeddings:
                    data.pop(
                        "embedding",
                        None,
                    )

                records.append(data)

            return {
                "dimension": self.dimension,
                "count": len(records),
                "records": records,
                "exported_at": _utc_now(),
            }

    def import_data(
        self,
        data: Dict[str, Any],
        replace: bool = False,
    ) -> int:
        """Import vectors from an exported structure."""

        if not isinstance(data, dict):
            raise ValueError(
                "Vector import data must be a dictionary."
            )

        imported = data.get(
            "records",
            [],
        )

        if not isinstance(imported, list):
            raise ValueError(
                "Vector records must be a list."
            )

        with self._lock:
            if replace:
                self._records.clear()

            count = 0

            for item in imported:
                if not isinstance(item, dict):
                    continue

                try:
                    record = VectorRecord.from_dict(
                        item
                    )

                    self._normalize_embedding(
                        record.embedding
                    )

                    self._records[
                        record.vector_id
                    ] = record

                    count += 1

                except Exception as exc:
                    logger.warning(
                        "Skipped invalid imported "
                        "vector: %s",
                        exc,
                    )

            self._enforce_limit()
            self._save_if_enabled()

            return count

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        with self._lock:
            records = list(
                self._records.values()
            )

            memory_types: Dict[str, int] = {}

            for record in records:
                memory_types[
                    record.memory_type
                ] = (
                    memory_types.get(
                        record.memory_type,
                        0,
                    )
                    + 1
                )

            return {
                "count": len(records),
                "dimension": self.dimension,
                "max_records": self.max_records,
                "archived": sum(
                    1
                    for record in records
                    if record.archived
                ),
                "active": sum(
                    1
                    for record in records
                    if not record.archived
                ),
                "memory_types": memory_types,
                "storage_path": str(
                    self.storage_path
                ),
            }


# ---------------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------------

LocalVectorStore = VectorStore
MemoryVectorStore = VectorStore


__all__ = [
    "VectorRecord",
    "VectorStore",
    "LocalVectorStore",
    "MemoryVectorStore",
]


