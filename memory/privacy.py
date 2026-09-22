"""
RENIX AI - Memory Privacy & Data Protection

Responsible for:
- Memory privacy controls
- Sensitive-memory detection
- Memory visibility levels
- User-controlled retention
- Consent checks
- Memory redaction
- Memory export/import helpers
- Secure deletion requests
- Privacy audit events
- Protected-memory handling
- Local-only memory policies

This module is intentionally independent from the actual database/vector
implementation. It provides policy and privacy decisions that the memory
manager can enforce.

Privacy levels:

    PUBLIC
    PRIVATE
    SENSITIVE
    HIGHLY_SENSITIVE
    SYSTEM

The default policy is conservative:
- Sensitive information is not automatically persisted unless explicitly
  allowed by the caller/policy.
- Protected memories cannot be deleted accidentally.
- Export operations can redact sensitive information.
- Privacy decisions are auditable.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


logger = logging.getLogger(
    "RENIX.memory.privacy"
)


# ============================================================================
# Utility functions
# ============================================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _utc_now().isoformat()


def _normalize(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(
        str(value)
        .strip()
        .lower()
        .split()
    )


def _hash_text(value: Any) -> str:
    return hashlib.sha256(
        _normalize(value).encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================================
# Privacy levels
# ============================================================================

class PrivacyLevel(str, Enum):
    """
    Privacy classification for a memory.
    """

    PUBLIC = "public"

    PRIVATE = "private"

    SENSITIVE = "sensitive"

    HIGHLY_SENSITIVE = (
        "highly_sensitive"
    )

    SYSTEM = "system"


# ============================================================================
# Privacy actions
# ============================================================================

class PrivacyAction(str, Enum):

    STORE = "store"

    RETRIEVE = "retrieve"

    EXPORT = "export"

    DELETE = "delete"

    MODIFY = "modify"

    SHARE = "share"


# ============================================================================
# Privacy decisions
# ============================================================================

@dataclass
class PrivacyDecision:
    """
    Result of a privacy policy evaluation.
    """

    allowed: bool

    action: str

    level: PrivacyLevel

    reason: str

    requires_confirmation: bool = False

    redaction_required: bool = False

    local_only: bool = False

    audit_required: bool = False

    matched_categories: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "allowed": self.allowed,
            "action": self.action,
            "level": self.level.value,
            "reason": self.reason,
            "requires_confirmation":
                self.requires_confirmation,
            "redaction_required":
                self.redaction_required,
            "local_only":
                self.local_only,
            "audit_required":
                self.audit_required,
            "matched_categories":
                list(
                    self.matched_categories
                ),
            "metadata":
                dict(self.metadata),
        }


# ============================================================================
# Privacy configuration
# ============================================================================

@dataclass
class PrivacyConfig:
    """
    RENIX privacy policy configuration.
    """

    enabled: bool = True

    allow_memory_storage: bool = True

    allow_sensitive_storage: bool = False

    allow_highly_sensitive_storage: bool = False

    allow_sensitive_retrieval: bool = False

    allow_highly_sensitive_retrieval: bool = False

    require_confirmation_for_sensitive: bool = True

    require_confirmation_for_delete: bool = True

    require_confirmation_for_export: bool = True

    redact_sensitive_exports: bool = True

    local_only_sensitive: bool = True

    local_only_highly_sensitive: bool = True

    audit_sensitive_operations: bool = True

    anonymize_logs: bool = True

    detect_sensitive_content: bool = True

    preserve_user_protected_memory: bool = True

    allow_memory_hashing: bool = True

    @classmethod
    def from_dict(
        cls,
        data: Optional[
            Dict[str, Any]
        ],
    ) -> "PrivacyConfig":

        config = cls()

        if not data:
            return config

        for key, value in data.items():

            if not hasattr(
                config,
                key,
            ):
                continue

            setattr(
                config,
                key,
                value,
            )

        return config


# ============================================================================
# Sensitive categories
# ============================================================================

@dataclass(frozen=True)
class SensitivePattern:
    category: str
    pattern: str
    level: PrivacyLevel


DEFAULT_SENSITIVE_PATTERNS = [
    SensitivePattern(
        category="password",
        pattern=r"\bpassword\s*[:=]\s*\S+",
        level=PrivacyLevel.HIGHLY_SENSITIVE,
    ),

    SensitivePattern(
        category="api_key",
        pattern=r"\b(?:api[_-]?key|secret[_-]?key)\s*[:=]\s*\S+",
        level=PrivacyLevel.HIGHLY_SENSITIVE,
    ),

    SensitivePattern(
        category="access_token",
        pattern=r"\b(?:access[_-]?token|auth[_-]?token)\s*[:=]\s*\S+",
        level=PrivacyLevel.HIGHLY_SENSITIVE,
    ),

    SensitivePattern(
        category="private_key",
        pattern=r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        level=PrivacyLevel.HIGHLY_SENSITIVE,
    ),

    SensitivePattern(
        category="credit_card",
        pattern=r"\b(?:\d[ -]*?){13,19}\b",
        level=PrivacyLevel.HIGHLY_SENSITIVE,
    ),

    SensitivePattern(
        category="email",
        pattern=r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        level=PrivacyLevel.SENSITIVE,
    ),

    SensitivePattern(
        category="phone",
        pattern=r"\b(?:+?\d[\d\s().-]{7,}\d)\b",
        level=PrivacyLevel.SENSITIVE,
    ),

    SensitivePattern(
        category="ip_address",
        pattern=r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        level=PrivacyLevel.SENSITIVE,
    ),
]


# ============================================================================
# Privacy audit event
# ============================================================================

@dataclass
class PrivacyAuditEvent:

    timestamp: str

    action: str

    allowed: bool

    privacy_level: str

    reason: str

    memory_id: Optional[str] = None

    categories: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "timestamp":
                self.timestamp,
            "action":
                self.action,
            "allowed":
                self.allowed,
            "privacy_level":
                self.privacy_level,
            "reason":
                self.reason,
            "memory_id":
                self.memory_id,
            "categories":
                list(self.categories),
            "metadata":
                dict(self.metadata),
        }


# ============================================================================
# Memory privacy metadata
# ============================================================================

@dataclass
class MemoryPrivacyMetadata:
    """
    Privacy metadata attached to an individual memory.
    """

    level: PrivacyLevel = (
        PrivacyLevel.PRIVATE
    )

    user_protected: bool = False

    local_only: bool = False

    consent_granted: bool = False

    retention_days: Optional[int] = None

    created_at: str = field(
        default_factory=_timestamp
    )

    categories: List[str] = field(
        default_factory=list
    )

    source: Optional[str] = None

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "privacy_level":
                self.level.value,
            "user_protected":
                self.user_protected,
            "local_only":
                self.local_only,
            "consent_granted":
                self.consent_granted,
            "retention_days":
                self.retention_days,
            "created_at":
                self.created_at,
            "categories":
                list(self.categories),
            "source":
                self.source,
        }


# ============================================================================
# Privacy manager
# ============================================================================

class PrivacyManager:
    """
    Central privacy policy engine for RENIX memory.

    It does not directly own the memory database.

    MemoryManager should call this class before:
        store
        retrieve
        export
        delete
        share
        modify
    """

    def __init__(
        self,
        config: Optional[
            PrivacyConfig
        ] = None,
        patterns: Optional[
            Sequence[
                SensitivePattern
            ]
        ] = None,
    ) -> None:

        self.config = (
            config
            or PrivacyConfig()
        )

        self.patterns = list(
            patterns
            or DEFAULT_SENSITIVE_PATTERNS
        )

        self._audit_log: List[
            PrivacyAuditEvent
        ] = []

        self._lock = threading.RLock()

    # ------------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------------

    def detect(
        self,
        text: Any,
    ) -> Dict[str, Any]:
        """
        Detect potentially sensitive content.

        Returns:
            {
                "sensitive": bool,
                "level": "...",
                "categories": [...]
            }
        """

        if not self.config.detect_sensitive_content:
            return {
                "sensitive": False,
                "level":
                    PrivacyLevel.PRIVATE.value,
                "categories": [],
            }

        content = str(
            text or ""
        )

        matched: Set[str] = set()

        highest_level = (
            PrivacyLevel.PRIVATE
        )

        priority = {
            PrivacyLevel.PUBLIC: 0,
            PrivacyLevel.PRIVATE: 1,
            PrivacyLevel.SENSITIVE: 2,
            PrivacyLevel.HIGHLY_SENSITIVE: 3,
            PrivacyLevel.SYSTEM: 4,
        }

        for rule in self.patterns:

            try:

                if re.search(
                    rule.pattern,
                    content,
                    flags=re.IGNORECASE,
                ):

                    matched.add(
                        rule.category
                    )

                    if (
                        priority[
                            rule.level
                        ]
                        > priority[
                            highest_level
                        ]
                    ):
                        highest_level = (
                            rule.level
                        )

            except re.error as exc:

                logger.warning(
                    "Invalid privacy pattern %s: %s",
                    rule.category,
                    exc,
                )

        return {
            "sensitive":
                bool(matched),

            "level":
                highest_level.value,

            "categories":
                sorted(matched),
        }

    # ------------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------------

    def classify(
        self,
        text: Any,
        requested_level: Optional[
            PrivacyLevel
        ] = None,
    ) -> MemoryPrivacyMetadata:

        detection = self.detect(
            text
        )

        detected_level = PrivacyLevel(
            detection["level"]
        )

        if requested_level is None:
            level = detected_level

        else:

            priority = {
                PrivacyLevel.PUBLIC: 0,
                PrivacyLevel.PRIVATE: 1,
                PrivacyLevel.SENSITIVE: 2,
                PrivacyLevel.HIGHLY_SENSITIVE: 3,
                PrivacyLevel.SYSTEM: 4,
            }

            level = (
                requested_level
                if priority[
                    requested_level
                ]
                >= priority[
                    detected_level
                ]
                else detected_level
            )

        local_only = (
            level
            == PrivacyLevel.HIGHLY_SENSITIVE
            and self.config
            .local_only_highly_sensitive
        )

        if (
            level
            == PrivacyLevel.SENSITIVE
            and self.config
            .local_only_sensitive
        ):
            local_only = True

        return MemoryPrivacyMetadata(
            level=level,
            local_only=local_only,
            categories=detection[
                "categories"
            ],
        )

    # ------------------------------------------------------------------------
    # Store policy
    # ------------------------------------------------------------------------

    def can_store(
        self,
        text: Any,
        *,
        consent: bool = False,
        requested_level: Optional[
            PrivacyLevel
        ] = None,
        memory_id: Optional[str] = None,
    ) -> PrivacyDecision:

        metadata = self.classify(
            text,
            requested_level,
        )

        level = metadata.level

        if not self.config.enabled:

            decision = PrivacyDecision(
                allowed=True,
                action=PrivacyAction.STORE.value,
                level=level,
                reason="Privacy policy disabled.",
                audit_required=False,
                local_only=metadata.local_only,
                matched_categories=metadata.categories,
            )

            return decision

        if not self.config.allow_memory_storage:

            decision = PrivacyDecision(
                allowed=False,
                action=PrivacyAction.STORE.value,
                level=level,
                reason="Memory storage is disabled.",
                matched_categories=metadata.categories,
            )

            self._audit(
                decision,
                memory_id,
            )

            return decision

        if (
            level
            == PrivacyLevel.HIGHLY_SENSITIVE
        ):

            if not self.config.allow_highly_sensitive_storage:

                decision = PrivacyDecision(
                    allowed=False,
                    action=PrivacyAction.STORE.value,
                    level=level,
                    reason=(
                        "Highly sensitive memory "
                        "storage is disabled."
                    ),
                    requires_confirmation=True,
                    local_only=True,
                    audit_required=True,
                    matched_categories=(
                        metadata.categories
                    ),
                )

                self._audit(
                    decision,
                    memory_id,
                )

                return decision

            if (
                not consent
                and self.config
                .require_confirmation_for_sensitive
            ):

                decision = PrivacyDecision(
                    allowed=False,
                    action=PrivacyAction.STORE.value,
                    level=level,
                    reason=(
                        "Explicit consent is required "
                        "for highly sensitive memory."
                    ),
                    requires_confirmation=True,
                    local_only=True,
                    audit_required=True,
                    matched_categories=(
                        metadata.categories
                    ),
                )

                self._audit(
                    decision,
                    memory_id,
                )

                return decision

        if (
            level
            == PrivacyLevel.SENSITIVE
        ):

            if not self.config.allow_sensitive_storage:

                decision = PrivacyDecision(
                    allowed=False,
                    action=PrivacyAction.STORE.value,
                    level=level,
                    reason=(
                        "Sensitive memory storage "
                        "is disabled."
                    ),
                    requires_confirmation=True,
                    local_only=metadata.local_only,
                    audit_required=True,
                    matched_categories=(
                        metadata.categories
                    ),
                )

                self._audit(
                    decision,
                    memory_id,
                )

                return decision

            if (
                not consent
                and self.config
                .require_confirmation_for_sensitive
            ):

                decision = PrivacyDecision(
                    allowed=False,
                    action=PrivacyAction.STORE.value,
                    level=level,
                    reason=(
                        "Explicit consent is required "
                        "for sensitive memory."
                    ),
                    requires_confirmation=True,
                    local_only=metadata.local_only,
                    audit_required=True,
                    matched_categories=(
                        metadata.categories
                    ),
                )

                self._audit(
                    decision,
                    memory_id,
                )

                return decision

        decision = PrivacyDecision(
            allowed=True,
            action=PrivacyAction.STORE.value,
            level=level,
            reason="Memory storage permitted.",
            local_only=metadata.local_only,
            audit_required=(
                self.config
                .audit_sensitive_operations
                and level
                in (
                    PrivacyLevel.SENSITIVE,
                    PrivacyLevel.HIGHLY_SENSITIVE,
                )
            ),
            matched_categories=(
                metadata.categories
            ),
        )

        self._audit(
            decision,
            memory_id,
        )

        return decision

    # ------------------------------------------------------------------------
    # Retrieval policy
    # ------------------------------------------------------------------------

    def can_retrieve(
        self,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        *,
        memory_id: Optional[str] = None,
        authorized: bool = False,
    ) -> PrivacyDecision:

        level = self._level_from_metadata(
            metadata
        )

        categories = self._categories_from_metadata(
            metadata
        )

        if (
            level
            == PrivacyLevel.HIGHLY_SENSITIVE
        ):

            if (
                not self.config
                .allow_highly_sensitive_retrieval
            ):

                decision = PrivacyDecision(
                    allowed=False,
                    action=PrivacyAction.RETRIEVE.value,
                    level=level,
                    reason=(
                        "Highly sensitive memory "
                        "retrieval is disabled."
                    ),
                    requires_confirmation=True,
                    local_only=True,
                    audit_required=True,
                    matched_categories=categories,
                )

                self._audit(
                    decision,
                    memory_id,
                )

                return decision

            if not authorized:

                decision = PrivacyDecision(
                    allowed=False,
                    action=PrivacyAction.RETRIEVE.value,
                    level=level,
                    reason=(
                        "Authorization is required "
                        "to retrieve highly sensitive memory."
                    ),
                    requires_confirmation=True,
                    local_only=True,
                    audit_required=True,
                    matched_categories=categories,
                )

                self._audit(
                    decision,
                    memory_id,
                )

                return decision

        if (
            level
            == PrivacyLevel.SENSITIVE
            and not self.config
            .allow_sensitive_retrieval
        ):

            decision = PrivacyDecision(
                allowed=False,
                action=PrivacyAction.RETRIEVE.value,
                level=level,
                reason=(
                    "Sensitive memory retrieval "
                    "is disabled."
                ),
                requires_confirmation=True,
                local_only=True,
                audit_required=True,
                matched_categories=categories,
            )

            self._audit(
                decision,
                memory_id,
            )

            return decision

        decision = PrivacyDecision(
            allowed=True,
            action=PrivacyAction.RETRIEVE.value,
            level=level,
            reason="Memory retrieval permitted.",
            local_only=(
                level
                in (
                    PrivacyLevel.SENSITIVE,
                    PrivacyLevel.HIGHLY_SENSITIVE,
                )
            ),
            audit_required=(
                self.config
                .audit_sensitive_operations
                and level
                in (
                    PrivacyLevel.SENSITIVE,
                    PrivacyLevel.HIGHLY_SENSITIVE,
                )
            ),
            matched_categories=categories,
        )

        self._audit(
            decision,
            memory_id,
        )

        return decision

    # ------------------------------------------------------------------------
    # Delete policy
    # ------------------------------------------------------------------------

    def can_delete(
        self,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        *,
        memory_id: Optional[str] = None,
        confirmed: bool = False,
    ) -> PrivacyDecision:

        level = self._level_from_metadata(
            metadata
        )

        categories = self._categories_from_metadata(
            metadata
        )

        protected = bool(
            metadata
            and (
                metadata.get(
                    "user_protected"
                )
                or metadata.get(
                    "protected"
                )
                or metadata.get(
                    "permanent"
                )
                or metadata.get(
                    "never_delete"
                )
            )
        )

        if (
            protected
            and self.config
            .preserve_user_protected_memory
        ):

            decision = PrivacyDecision(
                allowed=False,
                action=PrivacyAction.DELETE.value,
                level=level,
                reason=(
                    "Memory is explicitly protected "
                    "from deletion."
                ),
                requires_confirmation=True,
                audit_required=True,
                matched_categories=categories,
            )

            self._audit(
                decision,
                memory_id,
            )

            return decision

        if (
            self.config
            .require_confirmation_for_delete
            and not confirmed
        ):

            decision = PrivacyDecision(
                allowed=False,
                action=PrivacyAction.DELETE.value,
                level=level,
                reason=(
                    "Deletion requires explicit "
                    "confirmation."
                ),
                requires_confirmation=True,
                audit_required=True,
                matched_categories=categories,
            )

            self._audit(
                decision,
                memory_id,
            )

            return decision

        decision = PrivacyDecision(
            allowed=True,
            action=PrivacyAction.DELETE.value,
            level=level,
            reason="Memory deletion permitted.",
            audit_required=True,
            matched_categories=categories,
        )

        self._audit(
            decision,
            memory_id,
        )

        return decision

    # ------------------------------------------------------------------------
    # Export policy
    # ------------------------------------------------------------------------

    def can_export(
        self,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        *,
        memory_id: Optional[str] = None,
        confirmed: bool = False,
    ) -> PrivacyDecision:

        level = self._level_from_metadata(
            metadata
        )

        categories = self._categories_from_metadata(
            metadata
        )

        if (
            self.config
            .require_confirmation_for_export
            and not confirmed
        ):

            decision = PrivacyDecision(
                allowed=False,
                action=PrivacyAction.EXPORT.value,
                level=level,
                reason=(
                    "Memory export requires "
                    "explicit confirmation."
                ),
                requires_confirmation=True,
                audit_required=True,
                matched_categories=categories,
            )

            self._audit(
                decision,
                memory_id,
            )

            return decision

        redaction = (
            self.config
            .redact_sensitive_exports
            and level
            in (
                PrivacyLevel.SENSITIVE,
                PrivacyLevel.HIGHLY_SENSITIVE,
            )
        )

        decision = PrivacyDecision(
            allowed=True,
            action=PrivacyAction.EXPORT.value,
            level=level,
            reason="Memory export permitted.",
            redaction_required=redaction,
            audit_required=True,
            matched_categories=categories,
        )

        self._audit(
            decision,
            memory_id,
        )

        return decision

    # ------------------------------------------------------------------------
    # Share policy
    # ------------------------------------------------------------------------

    def can_share(
        self,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        *,
        memory_id: Optional[str] = None,
        confirmed: bool = False,
    ) -> PrivacyDecision:

        level = self._level_from_metadata(
            metadata
        )

        categories = self._categories_from_metadata(
            metadata
        )

        if level in (
            PrivacyLevel.SENSITIVE,
            PrivacyLevel.HIGHLY_SENSITIVE,
        ):

            if not confirmed:

                decision = PrivacyDecision(
                    allowed=False,
                    action=PrivacyAction.SHARE.value,
                    level=level,
                    reason=(
                        "Sensitive memory cannot be "
                        "shared without explicit "
                        "confirmation."
                    ),
                    requires_confirmation=True,
                    local_only=True,
                    audit_required=True,
                    matched_categories=categories,
                )

                self._audit(
                    decision,
                    memory_id,
                )

                return decision

        decision = PrivacyDecision(
            allowed=True,
            action=PrivacyAction.SHARE.value,
            level=level,
            reason="Memory sharing permitted.",
            audit_required=True,
            matched_categories=categories,
        )

        self._audit(
            decision,
            memory_id,
        )

        return decision

    # ------------------------------------------------------------------------
    # Redaction
    # ------------------------------------------------------------------------

    def redact(
        self,
        text: Any,
    ) -> str:
        """
        Redact detected sensitive values.

        This is intended for:
            exports
            logs
            diagnostics
            non-sensitive displays

        It does NOT modify the original stored memory.
        """

        content = str(
            text or ""
        )

        for rule in self.patterns:

            try:

                replacement = (
                    f"[REDACTED:{rule.category.upper()}]"
                )

                content = re.sub(
                    rule.pattern,
                    replacement,
                    content,
                    flags=re.IGNORECASE,
                )

            except re.error:
                continue

        return content

    # ------------------------------------------------------------------------
    # Safe export
    # ------------------------------------------------------------------------

    def sanitize_for_export(
        self,
        memory: Dict[str, Any],
        *,
        confirmed: bool = False,
    ) -> Optional[
        Dict[str, Any]
    ]:

        metadata = memory.get(
            "privacy",
            {}
        )

        memory_id = memory.get(
            "memory_id"
        )

        decision = self.can_export(
            metadata,
            memory_id=memory_id,
            confirmed=confirmed,
        )

        if not decision.allowed:
            return None

        result = dict(
            memory
        )

        if decision.redaction_required:

            if "text" in result:
                result["text"] = (
                    self.redact(
                        result["text"]
                    )
                )

            if "content" in result:
                result["content"] = (
                    self.redact(
                        result["content"]
                    )
                )

        # Never export internal security fields.
        for key in (
            "password",
            "api_key",
            "secret",
            "access_token",
            "private_key",
        ):
            result.pop(
                key,
                None,
            )

        return result

    # ------------------------------------------------------------------------
    # Privacy metadata helpers
    # ------------------------------------------------------------------------

    @staticmethod
    def _level_from_metadata(
        metadata: Optional[
            Dict[str, Any]
        ],
    ) -> PrivacyLevel:

        if not metadata:
            return PrivacyLevel.PRIVATE

        raw = (
            metadata.get(
                "privacy_level"
            )
            or metadata.get(
                "level"
            )
            or PrivacyLevel.PRIVATE.value
        )

        try:
            return PrivacyLevel(
                raw
            )
        except ValueError:
            return PrivacyLevel.PRIVATE

    @staticmethod
    def _categories_from_metadata(
        metadata: Optional[
            Dict[str, Any]
        ],
    ) -> List[str]:

        if not metadata:
            return []

        categories = (
            metadata.get(
                "categories"
            )
            or metadata.get(
                "sensitive_categories"
            )
            or []
        )

        if isinstance(
            categories,
            str,
        ):
            return [categories]

        return [
            str(item)
            for item in categories
        ]

    # ------------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------------

    def _audit(
        self,
        decision: PrivacyDecision,
        memory_id: Optional[str],
    ) -> None:

        if not (
            self.config.audit_sensitive_operations
            or decision.audit_required
        ):
            return

        event = PrivacyAuditEvent(
            timestamp=_timestamp(),
            action=decision.action,
            allowed=decision.allowed,
            privacy_level=(
                decision.level.value
            ),
            reason=decision.reason,
            memory_id=memory_id,
            categories=list(
                decision.matched_categories
            ),
            metadata={
                "local_only":
                    decision.local_only,
                "requires_confirmation":
                    decision.requires_confirmation,
            },
        )

        with self._lock:
            self._audit_log.append(
                event
            )

            # Prevent unbounded memory growth.
            if len(
                self._audit_log
            ) > 5000:

                self._audit_log = (
                    self._audit_log[-5000:]
                )

    def get_audit_log(
        self,
        limit: int = 100,
    ) -> List[
        Dict[str, Any]
    ]:

        limit = max(
            1,
            int(limit),
        )

        with self._lock:

            events = (
                self._audit_log[
                    -limit:
                ]
            )

            return [
                event.to_dict()
                for event in events
            ]

    def clear_audit_log(
        self,
    ) -> None:

        with self._lock:
            self._audit_log.clear()

    # ------------------------------------------------------------------------
    # Privacy summary
    # ------------------------------------------------------------------------

    def privacy_summary(
        self,
    ) -> Dict[str, Any]:

        return {
            "enabled":
                self.config.enabled,

            "memory_storage":
                self.config
                .allow_memory_storage,

            "sensitive_storage":
                self.config
                .allow_sensitive_storage,

            "highly_sensitive_storage":
                self.config
                .allow_highly_sensitive_storage,

            "sensitive_retrieval":
                self.config
                .allow_sensitive_retrieval,

            "highly_sensitive_retrieval":
                self.config
                .allow_highly_sensitive_retrieval,

            "sensitive_export_redaction":
                self.config
                .redact_sensitive_exports,

            "local_only_sensitive":
                self.config
                .local_only_sensitive,

            "local_only_highly_sensitive":
                self.config
                .local_only_highly_sensitive,

            "audit_enabled":
                self.config
                .audit_sensitive_operations,

            "audit_events":
                len(
                    self._audit_log
                ),
        }


# ============================================================================
# Singleton
# ============================================================================

_default_privacy_manager: Optional[
    PrivacyManager
] = None

_default_lock = threading.Lock()


def get_privacy_manager() -> PrivacyManager:
    """
    Return the shared RENIX privacy manager.
    """

    global _default_privacy_manager

    if _default_privacy_manager is None:

        with _default_lock:

            if _default_privacy_manager is None:

                _default_privacy_manager = (
                    PrivacyManager()
                )

    return _default_privacy_manager


# ============================================================================
# Convenience API
# ============================================================================

def classify_memory(
    text: Any,
) -> MemoryPrivacyMetadata:

    return get_privacy_manager().classify(
        text
    )


def detect_sensitive(
    text: Any,
) -> Dict[str, Any]:

    return get_privacy_manager().detect(
        text
    )


def can_store_memory(
    text: Any,
    *,
    consent: bool = False,
) -> PrivacyDecision:

    return get_privacy_manager().can_store(
        text,
        consent=consent,
    )


def redact_sensitive(
    text: Any,
) -> str:

    return get_privacy_manager().redact(
        text
    )


# ============================================================================
# Compatibility aliases
# ============================================================================

MemoryPrivacy = PrivacyManager
PrivacyEngine = PrivacyManager


__all__ = [
    "PrivacyLevel",
    "PrivacyAction",
    "PrivacyDecision",
    "PrivacyConfig",
    "SensitivePattern",
    "PrivacyAuditEvent",
    "MemoryPrivacyMetadata",
    "PrivacyManager",
    "MemoryPrivacy",
    "PrivacyEngine",
    "get_privacy_manager",
    "classify_memory",
    "detect_sensitive",
    "can_store_memory",
    "redact_sensitive",
]


