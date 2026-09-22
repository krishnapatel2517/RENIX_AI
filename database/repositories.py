"""
RENIX Database Repositories
===========================

Repository layer for RENIX database models.

Responsibilities:
- CRUD operations for BaseModel objects
- Query helpers
- Model serialization/deserialization
- Safe parameterized SQL
- Pagination
- Counting
- Existence checks

Repositories keep database-specific logic out of the rest
of the RENIX application.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Generic, Optional, Sequence, TypeVar

from .database import (
    Database,
    DatabaseError,
    get_database,
)
from .models import (
    BaseModel,
    get_model,
)

logger = logging.getLogger(__name__)

ModelT = TypeVar(
    "ModelT",
    bound=BaseModel,
)


# ============================================================
# EXCEPTIONS
# ============================================================


class RepositoryError(DatabaseError):
    """Base repository exception."""


class RepositoryValidationError(
    RepositoryError
):
    """Raised when a model fails validation."""


class RepositoryNotFoundError(
    RepositoryError
):
    """Raised when a requested model does not exist."""


# ============================================================
# BASE REPOSITORY
# ============================================================


class BaseRepository(
    Generic[ModelT]
):
    """
    Generic CRUD repository for a RENIX model.
    """

    model_class: type[ModelT]

    def __init__(
        self,
        model_class: type[ModelT],
        database: Database | None = None,
    ) -> None:

        if not issubclass(
            model_class,
            BaseModel,
        ):
            raise TypeError(
                "model_class must inherit "
                "from BaseModel."
            )

        self.model_class = model_class

        self.database = (
            database
            if database is not None
            else get_database()
        )

        self.table_name = (
            model_class.table_name
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_model(
        self,
        model: ModelT,
    ) -> None:

        if not isinstance(
            model,
            self.model_class,
        ):
            raise RepositoryValidationError(
                f"Expected "
                f"{self.model_class.__name__}, "
                f"got {type(model).__name__}."
            )

        try:
            model.validate()

        except ValueError as exc:

            raise RepositoryValidationError(
                str(exc)
            ) from exc

    # ========================================================
    # COLUMN SAFETY
    # ========================================================

    def _allowed_fields(self) -> set[str]:

        return set(
            self.model_class.fields
        )

    def _validate_field(
        self,
        field: str,
    ) -> None:

        if field not in self._allowed_fields():

            raise RepositoryError(
                f"Unknown field '{field}' "
                f"for {self.table_name}."
            )

    def _validate_fields(
        self,
        fields: Sequence[str],
    ) -> None:

        for field in fields:
            self._validate_field(
                field
            )

    # ========================================================
    # CREATE
    # ========================================================

    def create(
        self,
        model: ModelT,
    ) -> ModelT:

        self._validate_model(
            model
        )

        data = model.to_database_dict(
            include_id=False
        )

        row_id = self.database.insert(
            self.table_name,
            data,
        )

        return replace(
            model,
            id=row_id,
        )

    # ========================================================
    # CREATE MANY
    # ========================================================

    def create_many(
        self,
        models: Sequence[ModelT],
    ) -> list[ModelT]:

        if not models:
            return []

        for model in models:
            self._validate_model(
                model
            )

        first_data = (
            models[0].to_database_dict(
                include_id=False
            )
        )

        columns = list(
            first_data.keys()
        )

        column_sql = ", ".join(
            f'"{column}"'
            for column in columns
        )

        placeholders = ", ".join(
            "?"
            for _ in columns
        )

        sql = (
            f'INSERT INTO "{self.table_name}" '
            f"({column_sql}) "
            f"VALUES ({placeholders})"
        )

        values = []

        for model in models:

            data = (
                model.to_database_dict(
                    include_id=False
                )
            )

            values.append(
                [
                    data.get(column)
                    for column in columns
                ]
            )

        self.database.executemany(
            sql,
            values,
        )

        # SQLite's executemany row IDs are not
        # reliably exposed, so return copies without
        # attempting to guess their IDs.
        return [
            replace(model)
            for model in models
        ]

    # ========================================================
    # FIND BY ID
    # ========================================================

    def get(
        self,
        model_id: int,
    ) -> Optional[ModelT]:

        row = self.database.query_one(
            f'''
            SELECT *
            FROM "{self.table_name}"
            WHERE "id" = ?
            LIMIT 1
            ''',
            (model_id,),
        )

        if row is None:
            return None

        return self.model_class.from_dict(
            row
        )

    def get_required(
        self,
        model_id: int,
    ) -> ModelT:

        model = self.get(
            model_id
        )

        if model is None:

            raise RepositoryNotFoundError(
                f"{self.table_name} "
                f"record {model_id} was not found."
            )

        return model

    # ========================================================
    # FIND ONE
    # ========================================================

    def find_one(
        self,
        where: str,
        parameters: Sequence[Any] = (),
        *,
        order_by: Optional[str] = None,
    ) -> Optional[ModelT]:

        sql = (
            f'SELECT * FROM "{self.table_name}" '
            f"WHERE {where}"
        )

        if order_by:

            sql += (
                " ORDER BY "
                f"{self._safe_order_by(order_by)}"
            )

        sql += " LIMIT 1"

        row = self.database.query_one(
            sql,
            parameters,
        )

        if row is None:
            return None

        return self.model_class.from_dict(
            row
        )

    # ========================================================
    # FIND MANY
    # ========================================================

    def find_many(
        self,
        where: Optional[str] = None,
        parameters: Sequence[Any] = (),
        *,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[ModelT]:

        sql = (
            f'SELECT * FROM "{self.table_name}"'
        )

        if where:
            sql += f" WHERE {where}"

        if order_by:
            sql += (
                " ORDER BY "
                f"{self._safe_order_by(order_by)}"
            )

        if limit is not None:

            if limit < 0:
                raise RepositoryError(
                    "limit cannot be negative."
                )

            sql += f" LIMIT {int(limit)}"

        if offset is not None:

            if offset < 0:
                raise RepositoryError(
                    "offset cannot be negative."
                )

            if limit is None:
                sql += " LIMIT -1"

            sql += (
                f" OFFSET {int(offset)}"
            )

        result = self.database.query(
            sql,
            parameters,
        )

        return [
            self.model_class.from_dict(
                row
            )
            for row in result.rows
        ]

    # ========================================================
    # ALL
    # ========================================================

    def all(
        self,
        *,
        order_by: Optional[str] = None,
    ) -> list[ModelT]:

        return self.find_many(
            order_by=order_by
        )

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        model: ModelT,
    ) -> ModelT:

        self._validate_model(
            model
        )

        if model.id is None:

            raise RepositoryValidationError(
                "Cannot update a model "
                "without an ID."
            )

        model.update_timestamp()

        data = model.to_database_dict(
            include_id=False
        )

        self.database.update(
            self.table_name,
            data,
            '"id" = ?',
            (model.id,),
        )

        return model

    # ========================================================
    # PARTIAL UPDATE
    # ========================================================

    def update_fields(
        self,
        model_id: int,
        fields: dict[str, Any],
    ) -> Optional[ModelT]:

        if not fields:
            return self.get(
                model_id
            )

        self._validate_fields(
            list(fields.keys())
        )

        fields = dict(fields)

        if "updated_at" in self._allowed_fields():

            from .models import utc_now

            fields["updated_at"] = utc_now()

        self.database.update(
            self.table_name,
            fields,
            '"id" = ?',
            (model_id,),
        )

        return self.get(
            model_id
        )

    # ========================================================
    # DELETE
    # ========================================================

    def delete(
        self,
        model_id: int,
    ) -> bool:

        count = self.database.delete(
            self.table_name,
            '"id" = ?',
            (model_id,),
        )

        return count > 0

    # ========================================================
    # DELETE WHERE
    # ========================================================

    def delete_where(
        self,
        where: str,
        parameters: Sequence[Any] = (),
    ) -> int:

        return self.database.delete(
            self.table_name,
            where,
            parameters,
        )

    # ========================================================
    # COUNT
    # ========================================================

    def count(
        self,
        where: Optional[str] = None,
        parameters: Sequence[Any] = (),
    ) -> int:

        sql = (
            f'SELECT COUNT(*) AS count '
            f'FROM "{self.table_name}"'
        )

        if where:
            sql += f" WHERE {where}"

        row = self.database.query_one(
            sql,
            parameters,
        )

        if row is None:
            return 0

        return int(
            row["count"]
        )

    # ========================================================
    # EXISTS
    # ========================================================

    def exists(
        self,
        where: str,
        parameters: Sequence[Any] = (),
    ) -> bool:

        row = self.database.query_one(
            f'''
            SELECT 1 AS found
            FROM "{self.table_name}"
            WHERE {where}
            LIMIT 1
            ''',
            parameters,
        )

        return row is not None

    # ========================================================
    # PAGINATION
    # ========================================================

    def paginate(
        self,
        page: int = 1,
        page_size: int = 50,
        *,
        where: Optional[str] = None,
        parameters: Sequence[Any] = (),
        order_by: Optional[str] = None,
    ) -> dict[str, Any]:

        if page < 1:

            raise RepositoryError(
                "page must be at least 1."
            )

        if page_size < 1:

            raise RepositoryError(
                "page_size must be at least 1."
            )

        total = self.count(
            where,
            parameters,
        )

        offset = (
            (page - 1)
            * page_size
        )

        items = self.find_many(
            where,
            parameters,
            order_by=order_by,
            limit=page_size,
            offset=offset,
        )

        total_pages = (
            (
                total
                + page_size
                - 1
            )
            // page_size
            if total
            else 0
        )

        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "has_next": (
                page < total_pages
            ),
            "has_previous": (
                page > 1
                and total_pages > 0
            ),
        }

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        field: str,
        value: str,
        *,
        limit: int = 50,
    ) -> list[ModelT]:

        self._validate_field(
            field
        )

        return self.find_many(
            f'"{field}" LIKE ?',
            (f"%{value}%",),
            limit=limit,
        )

    # ========================================================
    # ORDER BY SAFETY
    # ========================================================

    def _safe_order_by(
        self,
        order_by: str,
    ) -> str:

        parts = order_by.strip().split()

        if not parts:
            raise RepositoryError(
                "order_by cannot be empty."
            )

        field = parts[0]

        self._validate_field(
            field
        )

        direction = ""

        if len(parts) > 1:

            direction_value = (
                parts[1].upper()
            )

            if direction_value not in {
                "ASC",
                "DESC",
            }:

                raise RepositoryError(
                    "order_by direction must be "
                    "ASC or DESC."
                )

            direction = (
                f" {direction_value}"
            )

        return (
            f'"{field}"{direction}'
        )

    # ========================================================
    # RAW SELECT
    # ========================================================

    def raw_query(
        self,
        sql: str,
        parameters: Sequence[Any] = (),
    ) -> list[dict[str, Any]]:

        return self.database.query(
            sql,
            parameters,
        ).rows


# ============================================================
# SPECIALIZED REPOSITORIES
# ============================================================


class MemoryRepository(
    BaseRepository[ModelT]
):
    """
    Repository helper for RENIX memory records.
    """

    def find_by_key(
        self,
        key: str,
    ) -> Optional[ModelT]:

        return self.find_one(
            '"key" = ?',
            (key,),
        )

    def find_category(
        self,
        category: str,
        *,
        limit: int = 100,
    ) -> list[ModelT]:

        return self.find_many(
            '"category" = ?',
            (category,),
            order_by="updated_at DESC",
            limit=limit,
        )


class ConversationRepository(
    BaseRepository[ModelT]
):
    """
    Repository helper for conversations.
    """

    def find_by_session_id(
        self,
        session_id: str,
    ) -> Optional[ModelT]:

        return self.find_one(
            '"session_id" = ?',
            (session_id,),
        )

    def active_conversations(
        self,
        *,
        limit: int = 100,
    ) -> list[ModelT]:

        return self.find_many(
            '"is_archived" = ?',
            (0,),
            order_by="updated_at DESC",
            limit=limit,
        )


class MessageRepository(
    BaseRepository[ModelT]
):
    """
    Repository helper for conversation messages.
    """

    def conversation_messages(
        self,
        conversation_id: int,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelT]:

        return self.find_many(
            '"conversation_id" = ?',
            (conversation_id,),
            order_by="created_at ASC",
            limit=limit,
            offset=offset,
        )

    def count_conversation_messages(
        self,
        conversation_id: int,
    ) -> int:

        return self.count(
            '"conversation_id" = ?',
            (conversation_id,),
        )


class SettingRepository(
    BaseRepository[ModelT]
):
    """
    Repository helper for application settings.
    """

    def get_by_key(
        self,
        key: str,
    ) -> Optional[ModelT]:

        return self.find_one(
            '"key" = ?',
            (key,),
        )

    def set_value(
        self,
        key: str,
        value: Any,
        *,
        category: str = "general",
    ) -> ModelT:

        existing = self.get_by_key(
            key
        )

        if existing is None:

            model = self.model_class(
                key=key,
                value=str(value),
                category=category,
            )

            return self.create(
                model
            )

        updated = self.update_fields(
            existing.id,
            {
                "value": str(value),
                "category": category,
            },
        )

        if updated is None:

            raise RepositoryNotFoundError(
                f"Setting '{key}' "
                "could not be updated."
            )

        return updated


# ============================================================
# REPOSITORY FACTORY
# ============================================================


def create_repository(
    model_class: type[ModelT],
    database: Database | None = None,
) -> BaseRepository[ModelT]:

    return BaseRepository(
        model_class,
        database,
    )


def create_repository_for_table(
    table_name: str,
    database: Database | None = None,
) -> BaseRepository:

    model_class = get_model(
        table_name
    )

    return BaseRepository(
        model_class,
        database,
    )


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "BaseRepository",
    "MemoryRepository",
    "ConversationRepository",
    "MessageRepository",
    "SettingRepository",
    "RepositoryError",
    "RepositoryValidationError",
    "RepositoryNotFoundError",
    "create_repository",
    "create_repository_for_table",
]


