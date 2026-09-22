"""
RENIX AI - Memory Cleanup Engine

Responsible for maintaining the RENIX memory system.

Features:
- Remove expired memories
- Remove archived memories
- Remove low-importance memories
- Remove duplicate memories
- Remove empty/corrupted records
- Enforce memory limits
- Preserve important memories
- Dry-run support
- Cleanup statistics
- Safe deletion
- Thread-safe operation

This module is designed to work with:
    memory_manager.py
    memory_store.py
    vector_store.py
    retrieval.py
    embeddings.py

IMPORTANT:
This module never blindly deletes important memories. By default,
high-importance memories are protected.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

try:
    from .vector_store import (
        VectorRecord,
        VectorStore,
    )
except ImportError:
    from vector_store import (
        VectorRecord,
        VectorStore,
    )


logger = logging.getLogger(
    "RENIX.memory.memory_cleanup"
)


# ============================================================================
# Utility functions
# ============================================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(
    value: Any,
) -> Optional[datetime]:
    """Safely convert a value into an aware UTC datetime."""

    if value is None:
        return None

    if isinstance(value, datetime):
        result = value

    else:
        try:
            result = datetime.fromisoformat(
                str(value).replace(
                    "Z",
                    "+00:00",
                )
            )
        except (
            ValueError,
            TypeError,
        ):
            return None

    if result.tzinfo is None:
        result = result.replace(
            tzinfo=timezone.utc
        )

    return result.astimezone(
        timezone.utc
    )


def _normalize_text(
    text: Any,
) -> str:
    """Normalize memory text for duplicate detection."""

    if text is None:
        return ""

    return " ".join(
        str(text)
        .lower()
        .strip()
        .split()
    )


def _text_hash(
    text: Any,
) -> str:
    """Generate a deterministic hash for memory text."""

    normalized = _normalize_text(
        text
    )

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MemoryCleanupConfig:
    """Configuration for RENIX memory maintenance."""

    enabled: bool = True

    # Age-based cleanup.
    max_age_days: int = 365

    # Memories below this importance can be cleaned when old.
    low_importance_threshold: float = 0.20

    # Memories at or above this value are protected.
    protected_importance_threshold: float = 0.80

    # Archived memories can be removed after this period.
    archived_max_age_days: int = 180

    # Empty memories are always candidates for removal.
    remove_empty_memories: bool = True

    # Duplicate cleanup.
    remove_duplicates: bool = True

    # Whether archived memories are eligible for deletion.
    remove_archived: bool = True

    # Maximum total records.
    max_records: Optional[int] = None

    # Keep this many records even when over the configured age.
    minimum_records_to_keep: int = 100

    # Never delete memories newer than this age unless invalid/duplicate.
    minimum_age_hours: float = 24.0

    # Dry-run mode.
    dry_run: bool = False

    @classmethod
    def from_dict(
        cls,
        data: Optional[
            Dict[str, Any]
        ],
    ) -> "MemoryCleanupConfig":

        config = cls()

        if not data:
            return config

        for key, value in data.items():

            if not hasattr(
                config,
                key,
            ):
                continue

            try:
                setattr(
                    config,
                    key,
                    value,
                )
            except Exception:
                continue

        config.max_age_days = max(
            1,
            int(
                config.max_age_days
            ),
        )

        config.archived_max_age_days = max(
            1,
            int(
                config.archived_max_age_days
            ),
        )

        config.minimum_records_to_keep = max(
            0,
            int(
                config.minimum_records_to_keep
            ),
        )

        config.minimum_age_hours = max(
            0.0,
            float(
                config.minimum_age_hours
            ),
        )

        config.low_importance_threshold = max(
            0.0,
            min(
                1.0,
                float(
                    config.low_importance_threshold
                ),
            ),
        )

        config.protected_importance_threshold = max(
            0.0,
            min(
                1.0,
                float(
                    config.protected_importance_threshold
                ),
            ),
        )

        return config


# ============================================================================
# Cleanup candidate
# ============================================================================

@dataclass
class CleanupCandidate:
    """Represents a memory selected for cleanup."""

    memory_id: Optional[str]

    vector_id: Optional[str]

    reason: str

    text: str

    importance: float = 0.0

    created_at: Optional[str] = None

    updated_at: Optional[str] = None

    archived: bool = False

    duplicate_of: Optional[str] = None

    age_days: Optional[float] = None

    protected: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "memory_id": self.memory_id,
            "vector_id": self.vector_id,
            "reason": self.reason,
            "text": self.text,
            "importance": self.importance,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "archived": self.archived,
            "duplicate_of": self.duplicate_of,
            "age_days": self.age_days,
            "protected": self.protected,
            "metadata": dict(
                self.metadata
            ),
        }


# ============================================================================
# Cleanup report
# ============================================================================

@dataclass
class CleanupReport:
    """Result of a RENIX memory cleanup operation."""

    started_at: str

    completed_at: Optional[str] = None

    dry_run: bool = False

    total_before: int = 0

    total_after: int = 0

    candidates: int = 0

    deleted: int = 0

    skipped: int = 0

    protected: int = 0

    invalid: int = 0

    duplicates: int = 0

    expired: int = 0

    archived: int = 0

    low_importance: int = 0

    capacity: int = 0

    errors: int = 0

    candidate_details: List[
        CleanupCandidate
    ] = field(
        default_factory=list
    )

    error_messages: List[str] = field(
        default_factory=list
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "dry_run": self.dry_run,
            "total_before": self.total_before,
            "total_after": self.total_after,
            "candidates": self.candidates,
            "deleted": self.deleted,
            "skipped": self.skipped,
            "protected": self.protected,
            "invalid": self.invalid,
            "duplicates": self.duplicates,
            "expired": self.expired,
            "archived": self.archived,
            "low_importance": self.low_importance,
            "capacity": self.capacity,
            "errors": self.errors,
            "candidate_details": [
                candidate.to_dict()
                for candidate
                in self.candidate_details
            ],
            "error_messages": list(
                self.error_messages
            ),
        }


# ============================================================================
# Memory Cleanup Engine
# ============================================================================

class MemoryCleanupEngine:
    """
    Main cleanup service for RENIX.

    Example:

        cleanup = MemoryCleanupEngine(
            vector_store=store
        )

        report = cleanup.cleanup()

    Dry run:

        report = cleanup.cleanup(
            dry_run=True
        )
    """

    def __init__(
        self,
        vector_store: Optional[
            VectorStore
        ] = None,
        config: Optional[
            MemoryCleanupConfig
        ] = None,
    ) -> None:

        self.vector_store = (
            vector_store
            or VectorStore()
        )

        self.config = (
            config
            or MemoryCleanupConfig()
        )

        self._lock = threading.RLock()

    # ------------------------------------------------------------------------
    # Main cleanup
    # ------------------------------------------------------------------------

    def cleanup(
        self,
        dry_run: Optional[
            bool
        ] = None,
    ) -> CleanupReport:

        started = _utc_now()

        if dry_run is None:
            dry_run = (
                self.config.dry_run
            )

        report = CleanupReport(
            started_at=started.isoformat(),
            dry_run=bool(dry_run),
        )

        with self._lock:

            try:
                records = (
                    self.vector_store.all(
                        include_archived=True
                    )
                )

            except Exception as exc:

                report.errors += 1

                report.error_messages.append(
                    str(exc)
                )

                report.completed_at = (
                    _utc_now().isoformat()
                )

                return report

            report.total_before = len(
                records
            )

            candidates = (
                self.find_candidates(
                    records=records
                )
            )

            report.candidates = len(
                candidates
            )

            self._populate_reason_counts(
                report,
                candidates,
            )

            report.candidate_details = (
                candidates
            )

            if dry_run:

                report.skipped = len(
                    candidates
                )

            else:

                for candidate in candidates:

                    try:
                        deleted = (
                            self._delete_candidate(
                                candidate
                            )
                        )

                        if deleted:
                            report.deleted += 1
                        else:
                            report.skipped += 1

                    except Exception as exc:

                        report.errors += 1

                        report.error_messages.append(
                            (
                                f"{candidate.memory_id}: "
                                f"{exc}"
                            )
                        )

            try:
                remaining = (
                    self.vector_store.all(
                        include_archived=True
                    )
                )

                report.total_after = len(
                    remaining
                )

            except Exception as exc:

                report.errors += 1

                report.error_messages.append(
                    str(exc)
                )

                report.total_after = (
                    max(
                        0,
                        report.total_before
                        - report.deleted,
                    )
                )

        report.completed_at = (
            _utc_now().isoformat()
        )

        return report

    # ------------------------------------------------------------------------
    # Candidate discovery
    # ------------------------------------------------------------------------

    def find_candidates(
        self,
        records: Optional[
            Sequence[VectorRecord]
        ] = None,
    ) -> List[CleanupCandidate]:

        if records is None:
            records = (
                self.vector_store.all(
                    include_archived=True
                )
            )

        records = list(
            records
        )

        candidates: List[
            CleanupCandidate
        ] = []

        protected_ids: Set[
            str
        ] = set()

        # First identify protected memories.
        for record in records:

            if self._is_protected(
                record
            ):
                identifier = (
                    record.memory_id
                    or record.vector_id
                )

                if identifier:
                    protected_ids.add(
                        identifier
                    )

        # Empty/corrupt memories.
        if self.config.remove_empty_memories:

            for record in records:

                if self._is_protected(
                    record
                ):
                    continue

                if self._is_invalid(
                    record
                ):

                    candidate = (
                        self._candidate(
                            record,
                            reason="invalid",
                        )
                    )

                    candidates.append(
                        candidate
                    )

        # Duplicate memories.
        if self.config.remove_duplicates:

            duplicate_candidates = (
                self._find_duplicates(
                    records
                )
            )

            for candidate in (
                duplicate_candidates
            ):

                identifier = (
                    candidate.memory_id
                    or candidate.vector_id
                )

                if (
                    identifier
                    and identifier
                    in protected_ids
                ):
                    continue

                candidates.append(
                    candidate
                )

        # Age-based cleanup.
        now = _utc_now()

        for record in records:

            identifier = (
                record.memory_id
                or record.vector_id
            )

            if (
                identifier
                and identifier
                in protected_ids
            ):
                continue

            if self._is_invalid(
                record
            ):
                continue

            age_days = (
                self._age_days(
                    record,
                    now,
                )
            )

            if age_days is None:
                continue

            if (
                age_days * 24
                < self.config
                .minimum_age_hours
            ):
                continue

            if (
                record.archived
                and self.config
                .remove_archived
                and age_days
                >= self.config
                .archived_max_age_days
            ):

                candidates.append(
                    self._candidate(
                        record,
                        reason="archived",
                        age_days=age_days,
                    )
                )

                continue

            if (
                age_days
                >= self.config.max_age_days
                and record.importance
                < self.config
                .low_importance_threshold
            ):

                candidates.append(
                    self._candidate(
                        record,
                        reason="expired",
                        age_days=age_days,
                    )
                )

                continue

            if (
                age_days
                >= self.config.max_age_days
                and record.importance
                < self.config
                .protected_importance_threshold
            ):

                candidates.append(
                    self._candidate(
                        record,
                        reason="low_importance",
                        age_days=age_days,
                    )
                )

        # Capacity-based cleanup.
        if (
            self.config.max_records
            is not None
        ):

            capacity_candidates = (
                self._capacity_candidates(
                    records,
                    existing=candidates,
                )
            )

            candidates.extend(
                capacity_candidates
            )

        # Remove duplicate candidate entries.
        candidates = (
            self._deduplicate_candidates(
                candidates
            )
        )

        return candidates

    # ------------------------------------------------------------------------
    # Protected memories
    # ------------------------------------------------------------------------

    def _is_protected(
        self,
        record: VectorRecord,
    ) -> bool:

        if (
            record.importance
            >= self.config
            .protected_importance_threshold
        ):
            return True

        metadata = (
            record.metadata
            or {}
        )

        if metadata.get(
            "protected"
        ) is True:
            return True

        if metadata.get(
            "permanent"
        ) is True:
            return True

        if metadata.get(
            "pinned"
        ) is True:
            return True

        if metadata.get(
            "keep"
        ) is True:
            return True

        if metadata.get(
            "never_delete"
        ) is True:
            return True

        return False

    # ------------------------------------------------------------------------
    # Invalid records
    # ------------------------------------------------------------------------

    @staticmethod
    def _is_invalid(
        record: VectorRecord,
    ) -> bool:

        text = str(
            record.text
            or ""
        ).strip()

        if not text:
            return True

        embedding = (
            getattr(
                record,
                "embedding",
                None,
            )
        )

        if not embedding:
            return True

        try:
            if not all(
                isinstance(
                    value,
                    (int, float),
                )
                for value in embedding
            ):
                return True
        except Exception:
            return True

        return False

    # ------------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------------

    def _find_duplicates(
        self,
        records: Sequence[
            VectorRecord
        ],
    ) -> List[CleanupCandidate]:

        seen: Dict[
            str,
            VectorRecord
        ] = {}

        candidates = []

        for record in records:

            if self._is_invalid(
                record
            ):
                continue

            normalized = _normalize_text(
                record.text
            )

            if not normalized:
                continue

            key = _text_hash(
                normalized
            )

            existing = seen.get(
                key
            )

            if existing is None:

                seen[key] = record
                continue

            # Keep the more important memory.
            if (
                record.importance
                > existing.importance
            ):

                duplicate = existing

                seen[key] = record

            elif (
                record.importance
                == existing.importance
            ):

                existing_time = (
                    _parse_datetime(
                        existing.updated_at
                    )
                    or _parse_datetime(
                        existing.created_at
                    )
                )

                current_time = (
                    _parse_datetime(
                        record.updated_at
                    )
                    or _parse_datetime(
                        record.created_at
                    )
                )

                if (
                    current_time
                    and existing_time
                    and current_time
                    > existing_time
                ):

                    duplicate = existing
                    seen[key] = record

                else:
                    duplicate = record

            else:
                duplicate = record

            if self._is_protected(
                duplicate
            ):
                continue

            keeper = seen[key]

            candidates.append(
                self._candidate(
                    duplicate,
                    reason="duplicate",
                    duplicate_of=(
                        keeper.memory_id
                        or keeper.vector_id
                    ),
                )
            )

        return candidates

    # ------------------------------------------------------------------------
    # Capacity management
    # ------------------------------------------------------------------------

    def _capacity_candidates(
        self,
        records: Sequence[
            VectorRecord
        ],
        existing: Sequence[
            CleanupCandidate
        ],
    ) -> List[CleanupCandidate]:

        max_records = (
            self.config.max_records
        )

        if max_records is None:
            return []

        max_records = max(
            0,
            int(max_records),
        )

        current_count = len(
            records
        )

        if current_count <= max_records:
            return []

        required = (
            current_count
            - max_records
        )

        existing_ids = {
            candidate.memory_id
            or candidate.vector_id
            for candidate in existing
        }

        candidates = []

        eligible = []

        for record in records:

            identifier = (
                record.memory_id
                or record.vector_id
            )

            if identifier in existing_ids:
                continue

            if self._is_protected(
                record
            ):
                continue

            if self._is_invalid(
                record
            ):
                eligible.append(
                    record
                )
                continue

            eligible.append(
                record
            )

        # Lowest importance first, then oldest.
        eligible.sort(
            key=lambda record: (
                float(
                    record.importance
                ),
                self._age_days(
                    record
                )
                or 0.0,
            ),
            reverse=False,
        )

        minimum_keep = (
            self.config
            .minimum_records_to_keep
        )

        allowed_deletions = max(
            0,
            current_count
            - minimum_keep,
        )

        required = min(
            required,
            allowed_deletions,
        )

        for record in eligible[
            :required
        ]:

            candidates.append(
                self._candidate(
                    record,
                    reason="capacity",
                    age_days=self._age_days(
                        record
                    ),
                )
            )

        return candidates

    # ------------------------------------------------------------------------
    # Candidate creation
    # ------------------------------------------------------------------------

    def _candidate(
        self,
        record: VectorRecord,
        reason: str,
        age_days: Optional[
            float
        ] = None,
        duplicate_of: Optional[
            str
        ] = None,
    ) -> CleanupCandidate:

        return CleanupCandidate(
            memory_id=record.memory_id,
            vector_id=record.vector_id,
            reason=reason,
            text=record.text,
            importance=float(
                record.importance
            ),
            created_at=record.created_at,
            updated_at=record.updated_at,
            archived=bool(
                record.archived
            ),
            duplicate_of=duplicate_of,
            age_days=age_days,
            protected=self._is_protected(
                record
            ),
            metadata=dict(
                record.metadata
                or {}
            ),
        )

    # ------------------------------------------------------------------------
    # Age calculation
    # ------------------------------------------------------------------------

    @staticmethod
    def _age_days(
        record: VectorRecord,
        now: Optional[
            datetime
        ] = None,
    ) -> Optional[float]:

        if now is None:
            now = _utc_now()

        timestamp = (
            _parse_datetime(
                record.updated_at
            )
            or _parse_datetime(
                record.created_at
            )
        )

        if timestamp is None:
            return None

        seconds = max(
            0.0,
            (
                now - timestamp
            ).total_seconds(),
        )

        return seconds / 86400.0

    # ------------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------------

    def _delete_candidate(
        self,
        candidate: CleanupCandidate,
    ) -> bool:

        if candidate.protected:
            return False

        identifier = (
            candidate.memory_id
            or candidate.vector_id
        )

        if not identifier:
            return False

        # Prefer memory ID when available.
        if candidate.memory_id:

            try:
                return bool(
                    self.vector_store.delete(
                        memory_id=(
                            candidate.memory_id
                        )
                    )
                )

            except TypeError:
                # Compatibility with stores that expose vector_id only.
                pass

        if candidate.vector_id:

            try:
                return bool(
                    self.vector_store.delete(
                        vector_id=(
                            candidate.vector_id
                        )
                    )
                )

            except TypeError:
                pass

        # Final compatibility attempt.
        try:
            return bool(
                self.vector_store.delete(
                    identifier
                )
            )
        except Exception:
            raise

    # ------------------------------------------------------------------------
    # Candidate deduplication
    # ------------------------------------------------------------------------

    @staticmethod
    def _deduplicate_candidates(
        candidates: Sequence[
            CleanupCandidate
        ],
    ) -> List[CleanupCandidate]:

        seen = set()
        result = []

        for candidate in candidates:

            identifier = (
                candidate.memory_id
                or candidate.vector_id
            )

            if not identifier:
                continue

            if identifier in seen:
                continue

            seen.add(
                identifier
            )

            result.append(
                candidate
            )

        return result

    # ------------------------------------------------------------------------
    # Report helpers
    # ------------------------------------------------------------------------

    @staticmethod
    def _populate_reason_counts(
        report: CleanupReport,
        candidates: Sequence[
            CleanupCandidate
        ],
    ) -> None:

        for candidate in candidates:

            reason = (
                candidate.reason
            )

            if reason == "invalid":
                report.invalid += 1

            elif reason == "duplicate":
                report.duplicates += 1

            elif reason == "expired":
                report.expired += 1

            elif reason == "archived":
                report.archived += 1

            elif reason == "low_importance":
                report.low_importance += 1

            elif reason == "capacity":
                report.capacity += 1

            if candidate.protected:
                report.protected += 1

    # ------------------------------------------------------------------------
    # Targeted cleanup
    # ------------------------------------------------------------------------

    def remove_duplicates(
        self,
        dry_run: Optional[
            bool
        ] = None,
    ) -> CleanupReport:

        records = (
            self.vector_store.all(
                include_archived=True
            )
        )

        candidates = (
            self._find_duplicates(
                records
            )
        )

        return self._execute_candidates(
            candidates,
            dry_run=dry_run,
        )

    def remove_expired(
        self,
        dry_run: Optional[
            bool
        ] = None,
    ) -> CleanupReport:

        records = (
            self.vector_store.all(
                include_archived=True
            )
        )

        candidates = []

        for record in records:

            if self._is_protected(
                record
            ):
                continue

            age = self._age_days(
                record
            )

            if age is None:
                continue

            if (
                age
                >= self.config.max_age_days
                and record.importance
                < self.config
                .protected_importance_threshold
            ):

                candidates.append(
                    self._candidate(
                        record,
                        reason="expired",
                        age_days=age,
                    )
                )

        return self._execute_candidates(
            candidates,
            dry_run=dry_run,
        )

    def remove_invalid(
        self,
        dry_run: Optional[
            bool
        ] = None,
    ) -> CleanupReport:

        records = (
            self.vector_store.all(
                include_archived=True
            )
        )

        candidates = [
            self._candidate(
                record,
                reason="invalid",
            )
            for record in records
            if (
                self._is_invalid(
                    record
                )
                and not self._is_protected(
                    record
                )
            )
        ]

        return self._execute_candidates(
            candidates,
            dry_run=dry_run,
        )

    # ------------------------------------------------------------------------
    # Candidate execution
    # ------------------------------------------------------------------------

    def _execute_candidates(
        self,
        candidates: Sequence[
            CleanupCandidate
        ],
        dry_run: Optional[
            bool
        ] = None,
    ) -> CleanupReport:

        if dry_run is None:
            dry_run = (
                self.config.dry_run
            )

        started = _utc_now()

        report = CleanupReport(
            started_at=started.isoformat(),
            dry_run=bool(dry_run),
            candidates=len(
                candidates
            ),
            candidate_details=list(
                candidates
            ),
        )

        try:
            report.total_before = len(
                self.vector_store.all(
                    include_archived=True
                )
            )
        except Exception:
            report.total_before = 0

        if dry_run:

            report.skipped = len(
                candidates
            )

        else:

            for candidate in candidates:

                if candidate.protected:
                    report.protected += 1
                    report.skipped += 1
                    continue

                try:
                    if self._delete_candidate(
                        candidate
                    ):
                        report.deleted += 1
                    else:
                        report.skipped += 1

                except Exception as exc:

                    report.errors += 1

                    report.error_messages.append(
                        (
                            f"{candidate.memory_id}: "
                            f"{exc}"
                        )
                    )

        try:
            report.total_after = len(
                self.vector_store.all(
                    include_archived=True
                )
            )
        except Exception:
            report.total_after = max(
                0,
                report.total_before
                - report.deleted,
            )

        self._populate_reason_counts(
            report,
            candidates,
        )

        report.completed_at = (
            _utc_now().isoformat()
        )

        return report

    # ------------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------------

    def preview(
        self,
    ) -> CleanupReport:
        """
        Preview cleanup without deleting anything.
        """

        return self.cleanup(
            dry_run=True
        )

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    def statistics(
        self,
    ) -> Dict[str, Any]:

        try:
            records = (
                self.vector_store.all(
                    include_archived=True
                )
            )

            total = len(
                records
            )

        except Exception:
            records = []
            total = 0

        protected = sum(
            1
            for record in records
            if self._is_protected(
                record
            )
        )

        invalid = sum(
            1
            for record in records
            if self._is_invalid(
                record
            )
        )

        return {
            "enabled":
                self.config.enabled,

            "total_records":
                total,

            "protected_records":
                protected,

            "invalid_records":
                invalid,

            "max_age_days":
                self.config.max_age_days,

            "archived_max_age_days":
                self.config
                .archived_max_age_days,

            "low_importance_threshold":
                self.config
                .low_importance_threshold,

            "protected_importance_threshold":
                self.config
                .protected_importance_threshold,

            "max_records":
                self.config.max_records,

            "minimum_records_to_keep":
                self.config
                .minimum_records_to_keep,

            "dry_run":
                self.config.dry_run,
        }


# ============================================================================
# Singleton
# ============================================================================

_default_cleanup_engine: Optional[
    MemoryCleanupEngine
] = None

_default_lock = threading.Lock()


def get_memory_cleanup() -> MemoryCleanupEngine:
    """Return the shared RENIX memory cleanup engine."""

    global _default_cleanup_engine

    if _default_cleanup_engine is None:

        with _default_lock:

            if _default_cleanup_engine is None:
                _default_cleanup_engine = (
                    MemoryCleanupEngine()
                )

    return _default_cleanup_engine


# ============================================================================
# Convenience functions
# ============================================================================

def cleanup_memory(
    dry_run: Optional[
        bool
    ] = None,
) -> CleanupReport:

    return get_memory_cleanup().cleanup(
        dry_run=dry_run
    )


def preview_cleanup() -> CleanupReport:

    return get_memory_cleanup().preview()


def cleanup_duplicates(
    dry_run: Optional[
        bool
    ] = None,
) -> CleanupReport:

    return get_memory_cleanup().remove_duplicates(
        dry_run=dry_run
    )


def cleanup_expired(
    dry_run: Optional[
        bool
    ] = None,
) -> CleanupReport:

    return get_memory_cleanup().remove_expired(
        dry_run=dry_run
    )


def cleanup_invalid(
    dry_run: Optional[
        bool
    ] = None,
) -> CleanupReport:

    return get_memory_cleanup().remove_invalid(
        dry_run=dry_run
    )


# ============================================================================
# Compatibility aliases
# ============================================================================

MemoryCleanup = MemoryCleanupEngine
MemoryCleaner = MemoryCleanupEngine
CleanupEngine = MemoryCleanupEngine


__all__ = [
    "MemoryCleanupConfig",
    "CleanupCandidate",
    "CleanupReport",
    "MemoryCleanupEngine",
    "MemoryCleanup",
    "MemoryCleaner",
    "CleanupEngine",
    "get_memory_cleanup",
    "cleanup_memory",
    "preview_cleanup",
    "cleanup_duplicates",
    "cleanup_expired",
    "cleanup_invalid",
]


