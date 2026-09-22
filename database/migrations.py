"""
RENIX Database Migrations
=========================

Schema migration manager for the RENIX SQLite database.

Responsibilities:
- Track schema versions
- Apply migrations sequentially
- Prevent skipped migrations
- Roll back the latest migration when supported
- Maintain migration history
- Provide migration status
- Keep migration logic separate from the database engine
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from .database import (
    Database,
    DatabaseError,
    get_database,
)

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================


class MigrationError(DatabaseError):
    """Base migration exception."""


class MigrationAlreadyAppliedError(
    MigrationError
):
    """Raised when a migration is already applied."""


class MigrationNotFoundError(
    MigrationError
):
    """Raised when a requested migration does not exist."""


class MigrationOrderError(
    MigrationError
):
    """Raised when migrations are out of order."""


class MigrationExecutionError(
    MigrationError
):
    """Raised when a migration fails."""


# ============================================================
# MIGRATION MODEL
# ============================================================


@dataclass(frozen=True)
class Migration:
    """
    Represents one database schema migration.
    """

    version: int

    name: str

    up: Callable[[Database], None]

    down: Optional[
        Callable[[Database], None]
    ] = None

    description: str = ""

    def validate(self) -> None:

        if self.version < 1:

            raise MigrationOrderError(
                "Migration version must be >= 1."
            )

        if not self.name.strip():

            raise MigrationError(
                "Migration name cannot be empty."
            )

        if not callable(self.up):

            raise MigrationError(
                "Migration 'up' must be callable."
            )

        if self.down is not None and not callable(
            self.down
        ):

            raise MigrationError(
                "Migration 'down' must be callable."
            )


# ============================================================
# MIGRATION MANAGER
# ============================================================


class MigrationManager:
    """
    Manages RENIX database schema migrations.
    """

    MIGRATION_TABLE = "schema_migrations"

    def __init__(
        self,
        database: Database | None = None,
    ) -> None:

        self.database = (
            database
            if database is not None
            else get_database()
        )

        self._migrations: dict[
            int,
            Migration,
        ] = {}

        self._history_table_ready = False

    # ========================================================
    # REGISTER
    # ========================================================

    def register(
        self,
        migration: Migration,
    ) -> None:

        migration.validate()

        if migration.version in self._migrations:

            raise MigrationError(
                f"Migration version "
                f"{migration.version} is already "
                "registered."
            )

        self._migrations[
            migration.version
        ] = migration

        logger.debug(
            "Registered migration %s: %s",
            migration.version,
            migration.name,
        )

    def register_many(
        self,
        migrations: list[Migration],
    ) -> None:

        for migration in migrations:
            self.register(
                migration
            )

    # ========================================================
    # SORTED MIGRATIONS
    # ========================================================

    def migrations(self) -> list[Migration]:

        return sorted(
            self._migrations.values(),
            key=lambda migration: migration.version,
        )

    # ========================================================
    # HISTORY TABLE
    # ========================================================

    def ensure_history_table(self) -> None:

        if self._history_table_ready:
            return

        self.database.execute(
            f"""
            CREATE TABLE IF NOT EXISTS
            "{self.MIGRATION_TABLE}" (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                applied_at TEXT NOT NULL
            )
            """
        )

        self._history_table_ready = True

    # ========================================================
    # APPLIED MIGRATIONS
    # ========================================================

    def applied_versions(self) -> list[int]:

        self.ensure_history_table()

        result = self.database.query(
            f"""
            SELECT version
            FROM "{self.MIGRATION_TABLE}"
            ORDER BY version ASC
            """
        )

        return [
            int(row["version"])
            for row in result.rows
        ]

    def applied_migrations(
        self,
    ) -> list[dict]:

        self.ensure_history_table()

        return self.database.query(
            f"""
            SELECT
                version,
                name,
                description,
                applied_at
            FROM "{self.MIGRATION_TABLE}"
            ORDER BY version ASC
            """
        ).rows

    def is_applied(
        self,
        version: int,
    ) -> bool:

        self.ensure_history_table()

        row = self.database.query_one(
            f"""
            SELECT 1
            FROM "{self.MIGRATION_TABLE}"
            WHERE version = ?
            LIMIT 1
            """,
            (version,),
        )

        return row is not None

    # ========================================================
    # CURRENT VERSION
    # ========================================================

    def current_version(self) -> int:

        versions = self.applied_versions()

        if not versions:
            return 0

        return max(
            versions
        )

    # ========================================================
    # VALIDATE MIGRATION CHAIN
    # ========================================================

    def validate_chain(self) -> None:

        migrations = self.migrations()

        expected = 1

        for migration in migrations:

            if migration.version != expected:

                raise MigrationOrderError(
                    "Migration chain contains a gap: "
                    f"expected version {expected}, "
                    f"found {migration.version}."
                )

            expected += 1

    # ========================================================
    # APPLY ONE
    # ========================================================

    def apply(
        self,
        version: int,
    ) -> None:

        self.ensure_history_table()

        migration = self._migrations.get(
            version
        )

        if migration is None:

            raise MigrationNotFoundError(
                f"Migration {version} "
                "is not registered."
            )

        if self.is_applied(
            version
        ):

            raise MigrationAlreadyAppliedError(
                f"Migration {version} "
                "has already been applied."
            )

        current = self.current_version()

        if version != current + 1:

            raise MigrationOrderError(
                f"Cannot apply migration {version}. "
                f"Current version is {current}; "
                f"expected {current + 1}."
            )

        try:

            with self.database.transaction():

                migration.up(
                    self.database
                )

                self.database.execute(
                    f"""
                    INSERT INTO
                    "{self.MIGRATION_TABLE}" (
                        version,
                        name,
                        description,
                        applied_at
                    )
                    VALUES (?, ?, ?, datetime('now'))
                    """,
                    (
                        migration.version,
                        migration.name,
                        migration.description,
                    ),
                    commit=False,
                )

                self.database.set_user_version(
                    migration.version
                )

            logger.info(
                "Applied migration %s: %s",
                migration.version,
                migration.name,
            )

        except Exception as exc:

            raise MigrationExecutionError(
                f"Migration {version} failed: "
                f"{exc}"
            ) from exc

    # ========================================================
    # APPLY ALL
    # ========================================================

    def migrate(
        self,
        target_version: Optional[int] = None,
    ) -> list[int]:

        self.ensure_history_table()

        self.validate_chain()

        current = self.current_version()

        available = [
            migration.version
            for migration in self.migrations()
        ]

        if target_version is None:

            target_version = (
                max(available)
                if available
                else current
            )

        if target_version < current:

            raise MigrationOrderError(
                f"Target version {target_version} "
                f"is below current version {current}. "
                "Use rollback() instead."
            )

        applied: list[int] = []

        for migration in self.migrations():

            if migration.version <= current:
                continue

            if migration.version > target_version:
                break

            self.apply(
                migration.version
            )

            applied.append(
                migration.version
            )

        return applied

    # ========================================================
    # ROLLBACK ONE
    # ========================================================

    def rollback(
        self,
        version: Optional[int] = None,
    ) -> int:

        self.ensure_history_table()

        current = self.current_version()

        if current == 0:

            raise MigrationError(
                "There are no migrations to roll back."
            )

        if version is None:
            version = current

        if version != current:

            raise MigrationOrderError(
                "Only the latest migration can "
                "be rolled back."
            )

        migration = self._migrations.get(
            version
        )

        if migration is None:

            raise MigrationNotFoundError(
                f"Migration {version} "
                "is not registered."
            )

        if migration.down is None:

            raise MigrationError(
                f"Migration {version} "
                "does not provide a rollback."
            )

        try:

            with self.database.transaction():

                migration.down(
                    self.database
                )

                self.database.execute(
                    f"""
                    DELETE FROM
                    "{self.MIGRATION_TABLE}"
                    WHERE version = ?
                    """,
                    (version,),
                    commit=False,
                )

                self.database.set_user_version(
                    version - 1
                )

            logger.info(
                "Rolled back migration %s: %s",
                migration.version,
                migration.name,
            )

            return version

        except Exception as exc:

            raise MigrationExecutionError(
                f"Rollback of migration "
                f"{version} failed: {exc}"
            ) from exc

    # ========================================================
    # ROLLBACK MANY
    # ========================================================

    def rollback_to(
        self,
        target_version: int,
    ) -> list[int]:

        if target_version < 0:

            raise MigrationOrderError(
                "Target version cannot be negative."
            )

        rolled_back: list[int] = []

        while (
            self.current_version()
            > target_version
        ):

            version = self.current_version()

            self.rollback(
                version
            )

            rolled_back.append(
                version
            )

        return rolled_back

    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> dict:

        self.ensure_history_table()

        registered = [
            {
                "version": migration.version,
                "name": migration.name,
                "description": migration.description,
                "applied": self.is_applied(
                    migration.version
                ),
            }
            for migration in self.migrations()
        ]

        current = self.current_version()

        latest = (
            max(
                self._migrations
            )
            if self._migrations
            else 0
        )

        return {
            "current_version": current,
            "latest_registered_version": latest,
            "pending_count": sum(
                1
                for item in registered
                if not item["applied"]
            ),
            "migrations": registered,
        }

    # ========================================================
    # PENDING
    # ========================================================

    def pending(
        self,
    ) -> list[Migration]:

        current = self.current_version()

        return [
            migration
            for migration in self.migrations()
            if migration.version > current
        ]

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
    ) -> list[int]:

        return self.rollback_to(
            0
        )


# ============================================================
# BUILT-IN INITIAL SCHEMA
# ============================================================


def migration_001_initial_schema_up(
    database: Database,
) -> None:
    """
    Initial RENIX schema.

    Tables are deliberately generic and can be extended by
    later migrations.
    """

    database.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL DEFAULT '',
            email TEXT,
            avatar_path TEXT,
            preferences TEXT,
            is_active INTEGER NOT NULL DEFAULT 1
        )
        """,
        commit=False,
    )

    database.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            key TEXT NOT NULL UNIQUE,
            content TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            importance REAL NOT NULL DEFAULT 0.5,
            source TEXT,
            metadata TEXT,
            is_active INTEGER NOT NULL DEFAULT 1
        )
        """,
        commit=False,
    )

    database.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            session_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL DEFAULT 'New Conversation',
            summary TEXT,
            metadata TEXT,
            is_archived INTEGER NOT NULL DEFAULT 0
        )
        """,
        commit=False,
    )

    database.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            conversation_id INTEGER,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model TEXT,
            metadata TEXT,
            token_count INTEGER,
            FOREIGN KEY (
                conversation_id
            )
            REFERENCES conversations(id)
            ON DELETE CASCADE
        )
        """,
        commit=False,
    )

    database.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            key TEXT NOT NULL UNIQUE,
            value TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            description TEXT,
            is_secret INTEGER NOT NULL DEFAULT 0
        )
        """,
        commit=False,
    )

    database.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            priority INTEGER NOT NULL DEFAULT 0,
            scheduled_at TEXT,
            completed_at TEXT,
            metadata TEXT
        )
        """,
        commit=False,
    )

    database.execute(
        """
        CREATE TABLE IF NOT EXISTS file_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            path TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            extension TEXT,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            checksum TEXT,
            mime_type TEXT,
            metadata TEXT
        )
        """,
        commit=False,
    )

    database.execute(
        """
        CREATE TABLE IF NOT EXISTS automations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            name TEXT NOT NULL,
            trigger_type TEXT NOT NULL DEFAULT 'manual',
            trigger_config TEXT,
            action_config TEXT,
            enabled INTEGER NOT NULL DEFAULT 1,
            last_run_at TEXT
        )
        """,
        commit=False,
    )

    # Indexes
    database.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_memories_category
        ON memories(category)
        """,
        commit=False,
    )

    database.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_messages_conversation
        ON messages(conversation_id, created_at)
        """,
        commit=False,
    )

    database.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_tasks_status
        ON tasks(status)
        """,
        commit=False,
    )

    database.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_file_records_checksum
        ON file_records(checksum)
        """,
        commit=False,
    )


def migration_001_initial_schema_down(
    database: Database,
) -> None:

    tables = [
        "automations",
        "file_records",
        "tasks",
        "messages",
        "conversations",
        "memories",
        "settings",
        "users",
    ]

    for table in tables:

        database.execute(
            f'DROP TABLE IF EXISTS "{table}"',
            commit=False,
        )


# ============================================================
# DEFAULT MIGRATION SET
# ============================================================


def create_default_migration_manager(
    database: Database | None = None,
) -> MigrationManager:

    manager = MigrationManager(
        database
    )

    manager.register(
        Migration(
            version=1,
            name="initial_schema",
            description=(
                "Create the initial RENIX "
                "database schema."
            ),
            up=migration_001_initial_schema_up,
            down=migration_001_initial_schema_down,
        )
    )

    return manager


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================


def migrate_database(
    database: Database | None = None,
) -> list[int]:

    manager = create_default_migration_manager(
        database
    )

    return manager.migrate()


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "Migration",
    "MigrationManager",
    "MigrationError",
    "MigrationAlreadyAppliedError",
    "MigrationNotFoundError",
    "MigrationOrderError",
    "MigrationExecutionError",
    "create_default_migration_manager",
    "migrate_database",
]


