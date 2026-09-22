"""
RENIX Database Models
=====================

Lightweight database model layer for RENIX.

This module provides reusable model primitives without forcing
the rest of RENIX to work directly with sqlite3.Row objects.

The models are intentionally dependency-light and compatible
with the Database engine in database.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional


# ============================================================
# HELPERS
# ============================================================


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _serialize_value(value: Any) -> Any:
    """Convert common Python values into database-friendly values."""

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, bool):
        return int(value)

    return value


# ============================================================
# BASE MODEL
# ============================================================


@dataclass
class BaseModel:
    """
    Base model used by RENIX database entities.

    Subclasses can override:
        table_name
        primary_key
        fields
    """

    id: Optional[int] = None

    created_at: Optional[str] = None

    updated_at: Optional[str] = None

    table_name: ClassVar[str] = ""

    primary_key: ClassVar[str] = "id"

    fields: ClassVar[tuple[str, ...]] = (
        "id",
        "created_at",
        "updated_at",
    )

    def __post_init__(self) -> None:
        now = utc_now()

        if self.created_at is None:
            self.created_at = now

        if self.updated_at is None:
            self.updated_at = now

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_dict(
        self,
        *,
        include_none: bool = False,
    ) -> dict[str, Any]:
        """
        Convert the model into a dictionary.
        """

        data = asdict(self)

        if not include_none:

            data = {
                key: value
                for key, value in data.items()
                if value is not None
            }

        return {
            key: _serialize_value(value)
            for key, value in data.items()
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "BaseModel":
        """
        Create a model from a dictionary.

        Unknown fields are ignored so database rows can contain
        metadata that a particular model does not use.
        """

        valid_fields = set(
            cls.fields
        )

        filtered = {
            key: value
            for key, value in data.items()
            if key in valid_fields
        }

        return cls(**filtered)

    # ========================================================
    # DATABASE DATA
    # ========================================================

    def to_database_dict(
        self,
        *,
        include_id: bool = False,
    ) -> dict[str, Any]:

        data = self.to_dict()

        if not include_id:
            data.pop(
                self.primary_key,
                None,
            )

        return data

    def update_timestamp(self) -> None:
        self.updated_at = utc_now()

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(self) -> None:
        """
        Base validation hook.

        Subclasses should override this method when required.
        """

        if not self.table_name:
            raise ValueError(
                f"{self.__class__.__name__} "
                "must define table_name."
            )

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:

        values = self.to_dict(
            include_none=True
        )

        rendered = ", ".join(
            f"{key}={value!r}"
            for key, value in values.items()
        )

        return (
            f"{self.__class__.__name__}"
            f"({rendered})"
        )


# ============================================================
# USER MODEL
# ============================================================


@dataclass
class UserModel(BaseModel):
    """
    RENIX user/profile model.
    """

    username: str = ""

    display_name: str = ""

    email: Optional[str] = None

    avatar_path: Optional[str] = None

    preferences: Optional[str] = None

    is_active: bool = True

    table_name: ClassVar[str] = "users"

    fields: ClassVar[tuple[str, ...]] = (
        "id",
        "created_at",
        "updated_at",
        "username",
        "display_name",
        "email",
        "avatar_path",
        "preferences",
        "is_active",
    )

    def validate(self) -> None:

        super().validate()

        if not self.username.strip():

            raise ValueError(
                "Username cannot be empty."
            )


# ============================================================
# MEMORY MODEL
# ============================================================


@dataclass
class MemoryModel(BaseModel):
    """
    Persistent RENIX memory record.

    The content field can contain ordinary text or serialized
    structured memory depending on the memory manager.
    """

    key: str = ""

    content: str = ""

    category: str = "general"

    importance: float = 0.5

    source: Optional[str] = None

    metadata: Optional[str] = None

    is_active: bool = True

    table_name: ClassVar[str] = "memories"

    fields: ClassVar[tuple[str, ...]] = (
        "id",
        "created_at",
        "updated_at",
        "key",
        "content",
        "category",
        "importance",
        "source",
        "metadata",
        "is_active",
    )

    def validate(self) -> None:

        super().validate()

        if not self.key.strip():

            raise ValueError(
                "Memory key cannot be empty."
            )

        if not self.content.strip():

            raise ValueError(
                "Memory content cannot be empty."
            )

        if not 0.0 <= self.importance <= 1.0:

            raise ValueError(
                "Memory importance must be between "
                "0.0 and 1.0."
            )


# ============================================================
# CONVERSATION MODEL
# ============================================================


@dataclass
class ConversationModel(BaseModel):
    """
    RENIX conversation/session model.
    """

    session_id: str = ""

    title: str = "New Conversation"

    summary: Optional[str] = None

    metadata: Optional[str] = None

    is_archived: bool = False

    table_name: ClassVar[str] = "conversations"

    fields: ClassVar[tuple[str, ...]] = (
        "id",
        "created_at",
        "updated_at",
        "session_id",
        "title",
        "summary",
        "metadata",
        "is_archived",
    )

    def validate(self) -> None:

        super().validate()

        if not self.session_id.strip():

            raise ValueError(
                "Conversation session_id "
                "cannot be empty."
            )


# ============================================================
# MESSAGE MODEL
# ============================================================


@dataclass
class MessageModel(BaseModel):
    """
    Individual conversation message.
    """

    conversation_id: Optional[int] = None

    role: str = "user"

    content: str = ""

    model: Optional[str] = None

    metadata: Optional[str] = None

    token_count: Optional[int] = None

    table_name: ClassVar[str] = "messages"

    fields: ClassVar[tuple[str, ...]] = (
        "id",
        "created_at",
        "updated_at",
        "conversation_id",
        "role",
        "content",
        "model",
        "metadata",
        "token_count",
    )

    VALID_ROLES: ClassVar[tuple[str, ...]] = (
        "system",
        "user",
        "assistant",
        "tool",
    )

    def validate(self) -> None:

        super().validate()

        if self.role not in self.VALID_ROLES:

            raise ValueError(
                f"Invalid message role: "
                f"{self.role}"
            )

        if not self.content.strip():

            raise ValueError(
                "Message content cannot be empty."
            )


# ============================================================
# SETTING MODEL
# ============================================================


@dataclass
class SettingModel(BaseModel):
    """
    Persistent RENIX application setting.
    """

    key: str = ""

    value: str = ""

    category: str = "general"

    description: Optional[str] = None

    is_secret: bool = False

    table_name: ClassVar[str] = "settings"

    fields: ClassVar[tuple[str, ...]] = (
        "id",
        "created_at",
        "updated_at",
        "key",
        "value",
        "category",
        "description",
        "is_secret",
    )

    def validate(self) -> None:

        super().validate()

        if not self.key.strip():

            raise ValueError(
                "Setting key cannot be empty."
            )


# ============================================================
# TASK MODEL
# ============================================================


@dataclass
class TaskModel(BaseModel):
    """
    RENIX automation/task record.
    """

    name: str = ""

    description: Optional[str] = None

    status: str = "pending"

    priority: int = 0

    scheduled_at: Optional[str] = None

    completed_at: Optional[str] = None

    metadata: Optional[str] = None

    table_name: ClassVar[str] = "tasks"

    fields: ClassVar[tuple[str, ...]] = (
        "id",
        "created_at",
        "updated_at",
        "name",
        "description",
        "status",
        "priority",
        "scheduled_at",
        "completed_at",
        "metadata",
    )

    VALID_STATUSES: ClassVar[tuple[str, ...]] = (
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
    )

    def validate(self) -> None:

        super().validate()

        if not self.name.strip():

            raise ValueError(
                "Task name cannot be empty."
            )

        if self.status not in self.VALID_STATUSES:

            raise ValueError(
                f"Invalid task status: "
                f"{self.status}"
            )


# ============================================================
# FILE RECORD MODEL
# ============================================================


@dataclass
class FileRecordModel(BaseModel):
    """
    RENIX file-system metadata record.

    This stores metadata only; it does not store the actual
    file contents.
    """

    path: str = ""

    name: str = ""

    extension: Optional[str] = None

    size_bytes: int = 0

    checksum: Optional[str] = None

    mime_type: Optional[str] = None

    metadata: Optional[str] = None

    table_name: ClassVar[str] = "file_records"

    fields: ClassVar[tuple[str, ...]] = (
        "id",
        "created_at",
        "updated_at",
        "path",
        "name",
        "extension",
        "size_bytes",
        "checksum",
        "mime_type",
        "metadata",
    )

    def validate(self) -> None:

        super().validate()

        if not self.path.strip():

            raise ValueError(
                "File path cannot be empty."
            )

        if self.size_bytes < 0:

            raise ValueError(
                "File size cannot be negative."
            )


# ============================================================
# AUTOMATION MODEL
# ============================================================


@dataclass
class AutomationModel(BaseModel):
    """
    RENIX automation definition.
    """

    name: str = ""

    trigger_type: str = "manual"

    trigger_config: Optional[str] = None

    action_config: Optional[str] = None

    enabled: bool = True

    last_run_at: Optional[str] = None

    table_name: ClassVar[str] = "automations"

    fields: ClassVar[tuple[str, ...]] = (
        "id",
        "created_at",
        "updated_at",
        "name",
        "trigger_type",
        "trigger_config",
        "action_config",
        "enabled",
        "last_run_at",
    )

    def validate(self) -> None:

        super().validate()

        if not self.name.strip():

            raise ValueError(
                "Automation name cannot be empty."
            )


# ============================================================
# MODEL REGISTRY
# ============================================================


MODEL_REGISTRY: dict[
    str,
    type[BaseModel],
] = {
    UserModel.table_name: UserModel,
    MemoryModel.table_name: MemoryModel,
    ConversationModel.table_name: ConversationModel,
    MessageModel.table_name: MessageModel,
    SettingModel.table_name: SettingModel,
    TaskModel.table_name: TaskModel,
    FileRecordModel.table_name: FileRecordModel,
    AutomationModel.table_name: AutomationModel,
}


def get_model(
    table_name: str,
) -> type[BaseModel]:
    """
    Return the registered model for a table.
    """

    try:

        return MODEL_REGISTRY[
            table_name
        ]

    except KeyError as exc:

        raise KeyError(
            f"No RENIX model registered for "
            f"table '{table_name}'."
        ) from exc


def register_model(
    model: type[BaseModel],
) -> None:
    """
    Register a custom RENIX model.
    """

    if not issubclass(
        model,
        BaseModel,
    ):
        raise TypeError(
            "Registered model must inherit "
            "from BaseModel."
        )

    if not model.table_name:

        raise ValueError(
            "Model must define table_name."
        )

    MODEL_REGISTRY[
        model.table_name
    ] = model


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "BaseModel",
    "UserModel",
    "MemoryModel",
    "ConversationModel",
    "MessageModel",
    "SettingModel",
    "TaskModel",
    "FileRecordModel",
    "AutomationModel",
    "MODEL_REGISTRY",
    "get_model",
    "register_model",
    "utc_now",
]


