"""
RENIX Database Engine
=====================

Central SQLite database engine for RENIX.

Responsibilities:
- Create/connect to the RENIX SQLite database
- Configure SQLite safely
- Execute SQL queries
- Execute parameterized statements
- Transactions
- Backup/restore helpers
- Database health checks
- Schema inspection
- Thread-local connections
- Clean shutdown

No RENIX feature should directly manage SQLite connections.
Use this module as the central database layer.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================


class DatabaseError(Exception):
    """Base RENIX database exception."""


class DatabaseConnectionError(DatabaseError):
    """Raised when a database connection cannot be created."""


class DatabaseQueryError(DatabaseError):
    """Raised when a SQL operation fails."""


class DatabaseTransactionError(DatabaseError):
    """Raised when a transaction fails."""


class DatabaseBackupError(DatabaseError):
    """Raised when a database backup fails."""


class DatabaseConfigurationError(DatabaseError):
    """Raised when database configuration is invalid."""


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class DatabaseConfig:
    """
    Configuration for the RENIX SQLite database.
    """

    path: str | os.PathLike[str] = "data/renix.db"

    timeout: float = 10.0

    check_same_thread: bool = False

    foreign_keys: bool = True

    journal_mode: str = "WAL"

    synchronous: str = "NORMAL"

    busy_timeout: int = 10_000

    cache_size: int = -20_000

    temp_store: str = "MEMORY"

    auto_create_parent: bool = True

    backup_directory: Optional[
        str | os.PathLike[str]
    ] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def resolved_path(self) -> Path:
        return Path(
            self.path
        ).expanduser().resolve()

    def validate(self) -> None:
        if self.timeout < 0:
            raise DatabaseConfigurationError(
                "Database timeout cannot be negative."
            )

        if self.busy_timeout < 0:
            raise DatabaseConfigurationError(
                "busy_timeout cannot be negative."
            )

        if self.journal_mode.upper() not in {
            "DELETE",
            "TRUNCATE",
            "PERSIST",
            "MEMORY",
            "WAL",
            "OFF",
        }:
            raise DatabaseConfigurationError(
                f"Unsupported journal mode: "
                f"{self.journal_mode}"
            )

        if self.synchronous.upper() not in {
            "OFF",
            "NORMAL",
            "FULL",
            "EXTRA",
        }:
            raise DatabaseConfigurationError(
                f"Unsupported synchronous mode: "
                f"{self.synchronous}"
            )

        if self.temp_store.upper() not in {
            "DEFAULT",
            "FILE",
            "MEMORY",
        }:
            raise DatabaseConfigurationError(
                f"Unsupported temp_store: "
                f"{self.temp_store}"
            )


# ============================================================
# QUERY RESULT
# ============================================================


@dataclass
class QueryResult:
    """
    Standardized result returned by database operations.
    """

    success: bool

    rows: list[dict[str, Any]] = field(
        default_factory=list
    )

    row_count: int = 0

    lastrowid: Optional[int] = None

    columns: list[str] = field(
        default_factory=list
    )

    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "rows": self.rows,
            "row_count": self.row_count,
            "lastrowid": self.lastrowid,
            "columns": self.columns,
            "error": self.error,
        }


# ============================================================
# DATABASE ENGINE
# ============================================================


class Database:
    """
    Central SQLite engine used by RENIX.

    Connections are stored per-thread to avoid accidentally
    sharing SQLite connection objects across threads.
    """

    APPLICATION_ID = 0x52454E58  # "RENX"

    SCHEMA_VERSION = 1

    def __init__(
        self,
        config: DatabaseConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else DatabaseConfig()
        )

        self.config.validate()

        self.path = self.config.resolved_path()

        self._local = threading.local()

        self._lock = threading.RLock()

        self._closed = False

        self._initialized = False

        logger.info(
            "RENIX Database initialized at %s",
            self.path,
        )

    # ========================================================
    # DIRECTORY
    # ========================================================

    def _ensure_parent_directory(self) -> None:

        parent = self.path.parent

        if parent.exists():
            return

        if not self.config.auto_create_parent:

            raise DatabaseConnectionError(
                f"Database directory does not exist: "
                f"{parent}"
            )

        try:

            parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        except OSError as exc:

            raise DatabaseConnectionError(
                f"Unable to create database directory: "
                f"{parent}"
            ) from exc

    # ========================================================
    # CONNECTION
    # ========================================================

    def connect(self) -> sqlite3.Connection:
        """
        Return the current thread's database connection.
        """

        if self._closed:

            raise DatabaseConnectionError(
                "Database has been closed."
            )

        existing = getattr(
            self._local,
            "connection",
            None,
        )

        if existing is not None:

            try:

                existing.execute(
                    "SELECT 1"
                )

                return existing

            except sqlite3.Error:

                try:
                    existing.close()
                except Exception:
                    pass

                self._local.connection = None

        self._ensure_parent_directory()

        try:

            connection = sqlite3.connect(
                str(self.path),
                timeout=self.config.timeout,
                check_same_thread=(
                    self.config.check_same_thread
                ),
            )

            connection.row_factory = (
                sqlite3.Row
            )

            self._configure_connection(
                connection
            )

            self._local.connection = connection

            return connection

        except sqlite3.Error as exc:

            raise DatabaseConnectionError(
                f"Unable to connect to database: "
                f"{exc}"
            ) from exc

    # ========================================================
    # CONNECTION CONFIGURATION
    # ========================================================

    def _configure_connection(
        self,
        connection: sqlite3.Connection,
    ) -> None:

        try:

            connection.execute(
                f"PRAGMA busy_timeout = "
                f"{int(self.config.busy_timeout)}"
            )

            connection.execute(
                f"PRAGMA cache_size = "
                f"{int(self.config.cache_size)}"
            )

            connection.execute(
                "PRAGMA temp_store = "
                f"{self.config.temp_store.upper()}"
            )

            connection.execute(
                "PRAGMA synchronous = "
                f"{self.config.synchronous.upper()}"
            )

            if self.config.foreign_keys:

                connection.execute(
                    "PRAGMA foreign_keys = ON"
                )

            else:

                connection.execute(
                    "PRAGMA foreign_keys = OFF"
                )

            # WAL is intentionally applied separately because
            # it returns a value rather than behaving like a
            # normal setter.
            connection.execute(
                "PRAGMA journal_mode = "
                f"{self.config.journal_mode.upper()}"
            )

            connection.execute(
                f"PRAGMA application_id = "
                f"{self.APPLICATION_ID}"
            )

        except sqlite3.Error as exc:

            raise DatabaseConnectionError(
                "Unable to configure SQLite connection."
            ) from exc

    # ========================================================
    # CONTEXT MANAGER
    # ========================================================

    @contextmanager
    def connection(
        self,
    ) -> Generator[
        sqlite3.Connection,
        None,
        None,
    ]:

        connection = self.connect()

        try:

            yield connection

        except Exception:

            raise

    # ========================================================
    # TRANSACTION
    # ========================================================

    @contextmanager
    def transaction(
        self,
    ) -> Generator[
        sqlite3.Connection,
        None,
        None,
    ]:

        connection = self.connect()

        try:

            connection.execute(
                "BEGIN"
            )

            yield connection

            connection.commit()

        except Exception as exc:

            try:
                connection.rollback()
            except Exception:
                pass

            raise DatabaseTransactionError(
                f"Transaction failed: {exc}"
            ) from exc

    # ========================================================
    # EXECUTE
    # ========================================================

    def execute(
        self,
        sql: str,
        parameters: Sequence[Any] = (),
        *,
        commit: bool = True,
    ) -> QueryResult:

        if not sql or not sql.strip():

            raise DatabaseQueryError(
                "SQL query cannot be empty."
            )

        connection = self.connect()

        try:

            cursor = connection.execute(
                sql,
                tuple(parameters),
            )

            if commit:
                connection.commit()

            columns = (
                [
                    description[0]
                    for description
                    in cursor.description
                ]
                if cursor.description
                else []
            )

            return QueryResult(
                success=True,
                row_count=cursor.rowcount,
                lastrowid=cursor.lastrowid,
                columns=columns,
            )

        except sqlite3.Error as exc:

            if commit:

                try:
                    connection.rollback()
                except Exception:
                    pass

            logger.exception(
                "Database execute failed."
            )

            raise DatabaseQueryError(
                f"SQL execution failed: {exc}"
            ) from exc

    # ========================================================
    # EXECUTEMANY
    # ========================================================

    def executemany(
        self,
        sql: str,
        parameters: Iterable[
            Sequence[Any]
        ],
        *,
        commit: bool = True,
    ) -> QueryResult:

        if not sql or not sql.strip():

            raise DatabaseQueryError(
                "SQL query cannot be empty."
            )

        connection = self.connect()

        try:

            cursor = connection.executemany(
                sql,
                parameters,
            )

            if commit:
                connection.commit()

            return QueryResult(
                success=True,
                row_count=cursor.rowcount,
                lastrowid=cursor.lastrowid,
            )

        except sqlite3.Error as exc:

            if commit:

                try:
                    connection.rollback()
                except Exception:
                    pass

            raise DatabaseQueryError(
                f"Batch SQL execution failed: "
                f"{exc}"
            ) from exc

    # ========================================================
    # QUERY
    # ========================================================

    def query(
        self,
        sql: str,
        parameters: Sequence[Any] = (),
    ) -> QueryResult:

        if not sql or not sql.strip():

            raise DatabaseQueryError(
                "SQL query cannot be empty."
            )

        connection = self.connect()

        try:

            cursor = connection.execute(
                sql,
                tuple(parameters),
            )

            rows = [
                dict(row)
                for row in cursor.fetchall()
            ]

            columns = [
                description[0]
                for description
                in cursor.description
            ] if cursor.description else []

            return QueryResult(
                success=True,
                rows=rows,
                row_count=len(rows),
                columns=columns,
            )

        except sqlite3.Error as exc:

            raise DatabaseQueryError(
                f"Database query failed: {exc}"
            ) from exc

    # ========================================================
    # QUERY ONE
    # ========================================================

    def query_one(
        self,
        sql: str,
        parameters: Sequence[Any] = (),
    ) -> Optional[dict[str, Any]]:

        result = self.query(
            sql,
            parameters,
        )

        if not result.rows:
            return None

        return result.rows[0]

    # ========================================================
    # INSERT
    # ========================================================

    def insert(
        self,
        table: str,
        data: dict[str, Any],
        *,
        commit: bool = True,
    ) -> int:

        if not table:
            raise DatabaseQueryError(
                "Table name cannot be empty."
            )

        if not data:
            raise DatabaseQueryError(
                "Insert data cannot be empty."
            )

        columns = list(
            data.keys()
        )

        placeholders = ", ".join(
            "?"
            for _ in columns
        )

        column_sql = ", ".join(
            f'"{column}"'
            for column in columns
        )

        sql = (
            f'INSERT INTO "{table}" '
            f"({column_sql}) "
            f"VALUES ({placeholders})"
        )

        result = self.execute(
            sql,
            [
                data[column]
                for column in columns
            ],
            commit=commit,
        )

        if result.lastrowid is None:

            raise DatabaseQueryError(
                "INSERT did not return a row ID."
            )

        return int(
            result.lastrowid
        )

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        table: str,
        data: dict[str, Any],
        where: str,
        parameters: Sequence[Any] = (),
        *,
        commit: bool = True,
    ) -> int:

        if not table:
            raise DatabaseQueryError(
                "Table name cannot be empty."
            )

        if not data:
            raise DatabaseQueryError(
                "Update data cannot be empty."
            )

        if not where or not where.strip():

            raise DatabaseQueryError(
                "UPDATE requires a WHERE clause."
            )

        assignments = ", ".join(
            f'"{column}" = ?'
            for column in data
        )

        sql = (
            f'UPDATE "{table}" '
            f"SET {assignments} "
            f"WHERE {where}"
        )

        values = list(
            data.values()
        )

        values.extend(
            parameters
        )

        result = self.execute(
            sql,
            values,
            commit=commit,
        )

        return result.row_count

    # ========================================================
    # DELETE
    # ========================================================

    def delete(
        self,
        table: str,
        where: str,
        parameters: Sequence[Any] = (),
        *,
        commit: bool = True,
    ) -> int:

        if not table:
            raise DatabaseQueryError(
                "Table name cannot be empty."
            )

        if not where or not where.strip():

            raise DatabaseQueryError(
                "DELETE requires a WHERE clause."
            )

        sql = (
            f'DELETE FROM "{table}" '
            f"WHERE {where}"
        )

        result = self.execute(
            sql,
            parameters,
            commit=commit,
        )

        return result.row_count

    # ========================================================
    # TABLE MANAGEMENT
    # ========================================================

    def table_exists(
        self,
        table: str,
    ) -> bool:

        result = self.query_one(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table'
              AND name = ?
            LIMIT 1
            """,
            (table,),
        )

        return result is not None

    def list_tables(self) -> list[str]:

        result = self.query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )

        return [
            row["name"]
            for row in result.rows
        ]

    # ========================================================
    # TABLE INFORMATION
    # ========================================================

    def table_info(
        self,
        table: str,
    ) -> list[dict[str, Any]]:

        return self.query(
            f'PRAGMA table_info("{table}")'
        ).rows

    # ========================================================
    # INDEX INFORMATION
    # ========================================================

    def list_indexes(
        self,
        table: Optional[str] = None,
    ) -> list[dict[str, Any]]:

        if table:

            return self.query(
                f'PRAGMA index_list("{table}")'
            ).rows

        return self.query(
            """
            SELECT name, tbl_name
            FROM sqlite_master
            WHERE type = 'index'
            ORDER BY name
            """
        ).rows

    # ========================================================
    # SCHEMA
    # ========================================================

    def get_schema(self) -> list[dict[str, Any]]:

        return self.query(
            """
            SELECT
                type,
                name,
                tbl_name,
                sql
            FROM sqlite_master
            WHERE sql IS NOT NULL
            ORDER BY type, name
            """
        ).rows

    # ========================================================
    # SCHEMA VERSION
    # ========================================================

    def get_user_version(self) -> int:

        result = self.query_one(
            "PRAGMA user_version"
        )

        if not result:
            return 0

        return int(
            next(iter(result.values()))
        )

    def set_user_version(
        self,
        version: int,
    ) -> None:

        if version < 0:
            raise DatabaseConfigurationError(
                "Schema version cannot be negative."
            )

        connection = self.connect()

        try:

            connection.execute(
                f"PRAGMA user_version = "
                f"{int(version)}"
            )

            connection.commit()

        except sqlite3.Error as exc:

            connection.rollback()

            raise DatabaseQueryError(
                f"Unable to update schema version: "
                f"{exc}"
            ) from exc

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def initialize(
        self,
        *,
        create_schema: bool = False,
        schema_sql: Optional[
            str
        ] = None,
    ) -> None:

        self.connect()

        if schema_sql:

            try:

                self.connect().executescript(
                    schema_sql
                )

                self.connect().commit()

            except sqlite3.Error as exc:

                try:
                    self.connect().rollback()
                except Exception:
                    pass

                raise DatabaseQueryError(
                    f"Schema initialization failed: "
                    f"{exc}"
                ) from exc

        self._initialized = True

        logger.info(
            "RENIX database initialized."
        )

    # ========================================================
    # INTEGRITY CHECK
    # ========================================================

    def integrity_check(self) -> dict[str, Any]:

        try:

            result = self.query(
                "PRAGMA integrity_check"
            )

            values = [
                list(row.values())[0]
                for row in result.rows
            ]

            healthy = (
                values == ["ok"]
            )

            return {
                "healthy": healthy,
                "result": values,
            }

        except DatabaseError as exc:

            return {
                "healthy": False,
                "result": [],
                "error": str(exc),
            }

    # ========================================================
    # FOREIGN KEY CHECK
    # ========================================================

    def foreign_key_check(self) -> dict[str, Any]:

        try:

            result = self.query(
                "PRAGMA foreign_key_check"
            )

            return {
                "healthy": (
                    len(result.rows) == 0
                ),
                "violations": result.rows,
            }

        except DatabaseError as exc:

            return {
                "healthy": False,
                "violations": [],
                "error": str(exc),
            }

    # ========================================================
    # SIZE
    # ========================================================

    def size_bytes(self) -> int:

        if not self.path.exists():
            return 0

        return self.path.stat().st_size

    # ========================================================
    # BACKUP
    # ========================================================

    def backup(
        self,
        destination: str | os.PathLike[str],
    ) -> Path:

        self._ensure_parent_directory()

        destination_path = Path(
            destination
        ).expanduser().resolve()

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        source_connection = self.connect()

        try:

            destination_connection = (
                sqlite3.connect(
                    str(destination_path)
                )
            )

            try:

                source_connection.backup(
                    destination_connection
                )

            finally:

                destination_connection.close()

            logger.info(
                "Database backup created: %s",
                destination_path,
            )

            return destination_path

        except sqlite3.Error as exc:

            raise DatabaseBackupError(
                f"Database backup failed: {exc}"
            ) from exc

    # ========================================================
    # FILE COPY BACKUP
    # ========================================================

    def backup_file(
        self,
        destination: str | os.PathLike[str],
    ) -> Path:

        destination_path = Path(
            destination
        ).expanduser().resolve()

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            self.connect()

            shutil.copy2(
                self.path,
                destination_path,
            )

            return destination_path

        except OSError as exc:

            raise DatabaseBackupError(
                f"Unable to copy database backup: "
                f"{exc}"
            ) from exc

    # ========================================================
    # VACUUM
    # ========================================================

    def vacuum(self) -> None:

        connection = self.connect()

        try:

            connection.execute(
                "VACUUM"
            )

            connection.commit()

        except sqlite3.Error as exc:

            raise DatabaseQueryError(
                f"VACUUM failed: {exc}"
            ) from exc

    # ========================================================
    # CHECKPOINT
    # ========================================================

    def checkpoint(
        self,
        mode: str = "PASSIVE",
    ) -> dict[str, Any]:

        mode = mode.upper()

        allowed = {
            "PASSIVE",
            "FULL",
            "RESTART",
            "TRUNCATE",
        }

        if mode not in allowed:

            raise DatabaseConfigurationError(
                f"Unsupported checkpoint mode: "
                f"{mode}"
            )

        result = self.query(
            f"PRAGMA wal_checkpoint({mode})"
        )

        return (
            result.rows[0]
            if result.rows
            else {}
        )

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(self) -> dict[str, Any]:

        try:

            self.connect()

            connection_ok = (
                self.query_one(
                    "SELECT 1 AS healthy"
                )
                is not None
            )

            integrity = (
                self.integrity_check()
            )

            foreign_keys = (
                self.foreign_key_check()
            )

            return {
                "healthy": (
                    connection_ok
                    and integrity["healthy"]
                    and foreign_keys["healthy"]
                ),
                "connection": connection_ok,
                "integrity": integrity,
                "foreign_keys": foreign_keys,
                "path": str(self.path),
                "size_bytes": self.size_bytes(),
                "schema_version": (
                    self.get_user_version()
                ),
            }

        except Exception as exc:

            return {
                "healthy": False,
                "connection": False,
                "error": str(exc),
            }

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "path": str(self.path),
            "exists": self.path.exists(),
            "size_bytes": self.size_bytes(),
            "initialized": self._initialized,
            "closed": self._closed,
            "schema_version": (
                self.get_user_version()
                if not self._closed
                else None
            ),
            "foreign_keys": (
                self.config.foreign_keys
            ),
            "journal_mode": (
                self.config.journal_mode
            ),
            "synchronous": (
                self.config.synchronous
            ),
        }

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:

        connection = getattr(
            self._local,
            "connection",
            None,
        )

        if connection is not None:

            try:

                connection.close()

            except Exception:

                logger.exception(
                    "Error while closing database."
                )

            finally:

                self._local.connection = None

        self._closed = True

        logger.info(
            "RENIX database closed."
        )

    def shutdown(self) -> None:
        self.close()


# ============================================================
# DEFAULT DATABASE
# ============================================================


_default_database: Optional[
    Database
] = None

_default_lock = threading.Lock()


def get_database(
    config: DatabaseConfig | None = None,
) -> Database:

    global _default_database

    with _default_lock:

        if (
            _default_database is None
            or _default_database._closed
        ):

            _default_database = Database(
                config
            )

        return _default_database


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def execute(
    sql: str,
    parameters: Sequence[Any] = (),
) -> QueryResult:

    return get_database().execute(
        sql,
        parameters,
    )


def query(
    sql: str,
    parameters: Sequence[Any] = (),
) -> QueryResult:

    return get_database().query(
        sql,
        parameters,
    )


def query_one(
    sql: str,
    parameters: Sequence[Any] = (),
) -> Optional[dict[str, Any]]:

    return get_database().query_one(
        sql,
        parameters,
    )


def insert(
    table: str,
    data: dict[str, Any],
) -> int:

    return get_database().insert(
        table,
        data,
    )


def update(
    table: str,
    data: dict[str, Any],
    where: str,
    parameters: Sequence[Any] = (),
) -> int:

    return get_database().update(
        table,
        data,
        where,
        parameters,
    )


def delete(
    table: str,
    where: str,
    parameters: Sequence[Any] = (),
) -> int:

    return get_database().delete(
        table,
        where,
        parameters,
    )


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "Database",
    "DatabaseConfig",
    "QueryResult",
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "DatabaseTransactionError",
    "DatabaseBackupError",
    "DatabaseConfigurationError",
    "get_database",
    "execute",
    "query",
    "query_one",
    "insert",
    "update",
    "delete",
]


