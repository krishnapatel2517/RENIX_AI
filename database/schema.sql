-- ============================================================
-- RENIX DATABASE SCHEMA
-- ============================================================
--
-- SQLite schema for RENIX AI.
--
-- This file mirrors migration 001 and can also be used when
-- creating a fresh database manually.
--
-- Foreign keys, WAL mode, and other SQLite runtime settings
-- are configured by database.py.
-- ============================================================


PRAGMA foreign_keys = ON;


-- ============================================================
-- USERS
-- ============================================================

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
);


-- ============================================================
-- MEMORIES
-- ============================================================

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
);


-- ============================================================
-- CONVERSATIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    created_at TEXT NOT NULL,

    updated_at TEXT NOT NULL,

    session_id TEXT NOT NULL UNIQUE,

    title TEXT NOT NULL DEFAULT 'New Conversation',

    summary TEXT,

    metadata TEXT,

    is_archived INTEGER NOT NULL DEFAULT 0
);


-- ============================================================
-- MESSAGES
-- ============================================================

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

    FOREIGN KEY (conversation_id)
        REFERENCES conversations(id)
        ON DELETE CASCADE
);


-- ============================================================
-- SETTINGS
-- ============================================================

CREATE TABLE IF NOT EXISTS settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    created_at TEXT NOT NULL,

    updated_at TEXT NOT NULL,

    key TEXT NOT NULL UNIQUE,

    value TEXT NOT NULL,

    category TEXT NOT NULL DEFAULT 'general',

    description TEXT,

    is_secret INTEGER NOT NULL DEFAULT 0
);


-- ============================================================
-- TASKS
-- ============================================================

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
);


-- ============================================================
-- FILE RECORDS
-- ============================================================

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
);


-- ============================================================
-- AUTOMATIONS
-- ============================================================

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
);


-- ============================================================
-- INDEXES
-- ============================================================


CREATE INDEX IF NOT EXISTS
idx_memories_category
ON memories(category);


CREATE INDEX IF NOT EXISTS
idx_memories_importance
ON memories(importance);


CREATE INDEX IF NOT EXISTS
idx_messages_conversation
ON messages(
    conversation_id,
    created_at
);


CREATE INDEX IF NOT EXISTS
idx_tasks_status
ON tasks(status);


CREATE INDEX IF NOT EXISTS
idx_tasks_scheduled
ON tasks(scheduled_at);


CREATE INDEX IF NOT EXISTS
idx_file_records_checksum
ON file_records(checksum);


CREATE INDEX IF NOT EXISTS
idx_file_records_extension
ON file_records(extension);


CREATE INDEX IF NOT EXISTS
idx_automations_enabled
ON automations(enabled);


CREATE INDEX IF NOT EXISTS
idx_automations_trigger
ON automations(trigger_type);


-- ============================================================
-- MIGRATION HISTORY
-- ============================================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,

    name TEXT NOT NULL,

    description TEXT,

    applied_at TEXT NOT NULL
);


-- ============================================================
-- DEFAULT RENIX SETTINGS
-- ============================================================

-- These are intentionally inserted only when the setting
-- does not already exist.

INSERT OR IGNORE INTO settings (
    created_at,
    updated_at,
    key,
    value,
    category,
    description,
    is_secret
)
VALUES (
    datetime('now'),
    datetime('now'),
    'renix.version',
    '1.0.0',
    'system',
    'RENIX application version.',
    0
);


INSERT OR IGNORE INTO settings (
    created_at,
    updated_at,
    key,
    value,
    category,
    description,
    is_secret
)
VALUES (
    datetime('now'),
    datetime('now'),
    'assistant.name',
    'RENIX',
    'assistant',
    'Name used by the RENIX assistant.',
    0
);


INSERT OR IGNORE INTO settings (
    created_at,
    updated_at,
    key,
    value,
    category,
    description,
    is_secret
)
VALUES (
    datetime('now'),
    datetime('now'),
    'assistant.voice_enabled',
    'true',
    'assistant',
    'Enable or disable voice output.',
    0
);


INSERT OR IGNORE INTO settings (
    created_at,
    updated_at,
    key,
    value,
    category,
    description,
    is_secret
)
VALUES (
    datetime('now'),
    datetime('now'),
    'assistant.wake_word_enabled',
    'true',
    'assistant',
    'Enable RENIX wake-word detection.',
    0
);


INSERT OR IGNORE INTO settings (
    created_at,
    updated_at,
    key,
    value,
    category,
    description,
    is_secret
)
VALUES (
    datetime('now'),
    datetime('now'),
    'ui.theme',
    'holographic',
    'ui',
    'Default RENIX interface theme.',
    0
);


INSERT OR IGNORE INTO settings (
    created_at,
    updated_at,
    key,
    value,
    category,
    description,
    is_secret
)
VALUES (
    datetime('now'),
    datetime('now'),
    'ui.accent',
    'green',
    'ui',
    'Default RENIX holographic accent.',
    0
);


-- ============================================================
-- END OF SCHEMA
-- ============================================================
