"""
RENIX Database Package
======================

Database layer for RENIX.

This package provides the public database API while keeping
database implementation details inside the database module.

Main components:
    - Database
    - DatabaseConfig
    - DatabaseError
    - Repository layer
    - Database models
    - Migration support

The package is intentionally lightweight so other RENIX modules
can safely import it without creating a database connection
during module import.
"""

from __future__ import annotations

from typing import Any


# ============================================================
# PACKAGE VERSION
# ============================================================

__version__ = "1.0.0"
__author__ = "RENIX"
__package_name__ = "renix.database"


# ============================================================
# SAFE OPTIONAL IMPORTS
# ============================================================

# Database implementation
try:
    from .database import (
        Database,
        DatabaseConfig,
        DatabaseError,
    )
except ImportError:
    Database = None  # type: ignore[assignment]
    DatabaseConfig = None  # type: ignore[assignment]
    DatabaseError = None  # type: ignore[assignment]


# Models
try:
    from .models import (
        BaseModel,
    )
except ImportError:
    BaseModel = None  # type: ignore[assignment]


# Repositories
try:
    from .repositories import (
        BaseRepository,
    )
except ImportError:
    BaseRepository = None  # type: ignore[assignment]


# Migrations
try:
    from .migrations import (
        MigrationManager,
    )
except ImportError:
    MigrationManager = None  # type: ignore[assignment]


# ============================================================
# DATABASE PACKAGE STATUS
# ============================================================

def get_status() -> dict[str, Any]:
    """
    Return the availability of the RENIX database components.

    This function does not create a database connection.
    """

    return {
        "package": __package_name__,
        "version": __version__,
        "database_available": Database is not None,
        "models_available": BaseModel is not None,
        "repositories_available": BaseRepository is not None,
        "migrations_available": MigrationManager is not None,
    }


# ============================================================
# DATABASE FACTORY
# ============================================================

def create_database(
    *args: Any,
    **kwargs: Any,
):
    """
    Create and return a RENIX Database instance.

    Imports are intentionally resolved lazily so importing
    `renix.database` does not automatically initialize SQLite
    or perform filesystem operations.
    """

    if Database is None:
        raise RuntimeError(
            "RENIX Database implementation is unavailable. "
            "Check database/database.py and its dependencies."
        )

    return Database(
        *args,
        **kwargs,
    )


# ============================================================
# MIGRATION FACTORY
# ============================================================

def create_migration_manager(
    *args: Any,
    **kwargs: Any,
):
    """
    Create and return a migration manager.
    """

    if MigrationManager is None:
        raise RuntimeError(
            "RENIX MigrationManager is unavailable. "
            "Check database/migrations.py."
        )

    return MigrationManager(
        *args,
        **kwargs,
    )


# ============================================================
# HEALTH CHECK
# ============================================================

def health_check() -> dict[str, Any]:
    """
    Perform a lightweight package-level database health check.

    This checks whether the required database components can be
    imported. It does not modify data and does not automatically
    create a persistent database.
    """

    status = get_status()

    components = [
        status["database_available"],
        status["models_available"],
        status["repositories_available"],
        status["migrations_available"],
    ]

    healthy = all(components)

    return {
        "healthy": healthy,
        "status": status,
    }


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "Database",
    "DatabaseConfig",
    "DatabaseError",
    "BaseModel",
    "BaseRepository",
    "MigrationManager",
    "create_database",
    "create_migration_manager",
    "get_status",
    "health_check",
    "__version__",
]


