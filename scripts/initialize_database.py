from __future__ import annotations

import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATABASE_DIR = ROOT / "data"
DATABASE_FILE = DATABASE_DIR / "renix.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    priority TEXT NOT NULL DEFAULT 'normal',
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    payload TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    actor TEXT,
    status TEXT NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL
);
"""


def initialize_database() -> bool:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        connection = sqlite3.connect(DATABASE_FILE)

        try:
            connection.executescript(SCHEMA)
            connection.commit()
        finally:
            connection.close()

        print(f"[OK] Database initialized: {DATABASE_FILE}")
        return True

    except sqlite3.Error as exc:
        print(f"[ERROR] Database initialization failed: {exc}")
        return False


def main() -> int:
    print("=" * 70)
    print("RENIX DATABASE INITIALIZER")
    print("=" * 70)

    return 0 if initialize_database() else 1


if __name__ == "__main__":
    raise SystemExit(main())


