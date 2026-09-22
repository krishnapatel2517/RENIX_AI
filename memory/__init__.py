"""
RENIX AI - Memory Package

Central package for RENIX's persistent and contextual memory system.

Memory layers:
    - Short-term memory
    - Long-term memory
    - Episodic memory
    - Semantic memory
    - Procedural memory
    - Project memory
    - Task memory
    - Preference memory
    - Conversation memory

Supporting systems:
    - Memory manager
    - Memory store
    - Vector store
    - Embeddings
    - Retrieval
    - Memory search
    - Memory cleanup
    - Privacy controls

This package exposes the public memory API while keeping the individual
implementations modular.
"""

from .memory_manager import (
    MemoryManager,
)

from .memory_store import (
    MemoryStore,
)

from .short_term import (
    ShortTermMemory,
)

from .long_term import (
    LongTermMemory,
)

from .episodic import (
    EpisodicMemory,
)

from .semantic import (
    SemanticMemory,
)

from .procedural import (
    ProceduralMemory,
)

from .project_memory import (
    ProjectMemory,
)

from .task_memory import (
    TaskMemory,
)

from .preference_memory import (
    PreferenceMemory,
)

from .conversation_memory import (
    ConversationMemory,
)

from .vector_store import (
    VectorStore,
)

from .embeddings import (
    EmbeddingManager,
)

from .retrieval import (
    MemoryRetriever,
)

from .memory_search import (
    MemorySearch,
)

from .memory_cleanup import (
    MemoryCleanup,
)

from .privacy import (
    PrivacyManager,
    PrivacyConfig,
    PrivacyLevel,
    PrivacyAction,
    PrivacyDecision,
    MemoryPrivacyMetadata,
    PrivacyAuditEvent,
    SensitivePattern,
    get_privacy_manager,
    classify_memory,
    detect_sensitive,
    can_store_memory,
    redact_sensitive,
)


__all__ = [
    # Core
    "MemoryManager",
    "MemoryStore",

    # Memory layers
    "ShortTermMemory",
    "LongTermMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "ProjectMemory",
    "TaskMemory",
    "PreferenceMemory",
    "ConversationMemory",

    # Vector / retrieval
    "VectorStore",
    "EmbeddingManager",
    "MemoryRetriever",
    "MemorySearch",

    # Maintenance
    "MemoryCleanup",

    # Privacy
    "PrivacyManager",
    "PrivacyConfig",
    "PrivacyLevel",
    "PrivacyAction",
    "PrivacyDecision",
    "MemoryPrivacyMetadata",
    "PrivacyAuditEvent",
    "SensitivePattern",
    "get_privacy_manager",
    "classify_memory",
    "detect_sensitive",
    "can_store_memory",
    "redact_sensitive",
]


__version__ = "1.0.0"
__author__ = "RENIX AI"
__description__ = (
    "Persistent, contextual and privacy-aware "
    "memory system for RENIX AI."
)


