"""
RENIX AI - Preference Memory

Stores persistent user preferences.

Examples:
- Preferred response style
- Preferred language
- UI preferences
- Voice preferences
- Study preferences
- Coding preferences
- Notification preferences
- Frequently used settings
- User-defined preferences

Preference memory is different from normal conversation memory:
preferences represent relatively persistent information that should
influence RENIX's future behavior.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


logger = logging.getLogger(
    "RENIX.memory.preference_memory"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Preference:
    """Represents one persistent RENIX user preference."""

    preference_id: str

    key: str

    value: Any

    category: str = "general"

    confidence: float = 1.0

    source: str = "user"

    created_at: str = field(
        default_factory=_now
    )

    updated_at: str = field(
        default_factory=_now
    )

    usage_count: int = 0

    enabled: bool = True

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "preference_id":
                self.preference_id,
            "key":
                self.key,
            "value":
                self.value,
            "category":
                self.category,
            "confidence":
                self.confidence,
            "source":
                self.source,
            "created_at":
                self.created_at,
            "updated_at":
                self.updated_at,
            "usage_count":
                self.usage_count,
            "enabled":
                self.enabled,
            "metadata":
                self.metadata,
        }


class PreferenceMemory:
    """
    Persistent preference layer for RENIX.

    Preferences are indexed by category + key and can be retrieved
    individually or as a complete preference profile.
    """

    def __init__(
        self,
        store: Optional[Any] = None,
    ) -> None:

        self.store = store

        self._preferences: Dict[
            str,
            Preference,
        ] = {}

        self._lock = threading.RLock()

        logger.info(
            "PreferenceMemory initialized."
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> bool:

        if self.store is not None:

            method = getattr(
                self.store,
                "initialize",
                None,
            )

            if callable(method):

                try:
                    method()

                except Exception:
                    logger.exception(
                        "Preference store initialization failed."
                    )

                    return False

        return True

    # ------------------------------------------------------------------
    # Set preference
    # ------------------------------------------------------------------

    def set(
        self,
        key: str,
        value: Any,
        *,
        category: str = "general",
        confidence: float = 1.0,
        source: str = "user",
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> str:

        key = str(
            key
        ).strip()

        if not key:
            raise ValueError(
                "Preference key cannot be empty."
            )

        preference_key = self._make_key(
            category,
            key,
        )

        confidence = self._clamp(
            confidence
        )

        with self._lock:

            existing = (
                self._preferences.get(
                    preference_key
                )
            )

            if existing is not None:

                existing.value = value

                existing.confidence = (
                    confidence
                )

                existing.source = source

                existing.updated_at = _now()

                if metadata:
                    existing.metadata.update(
                        metadata
                    )

                preference = existing

            else:

                preference = Preference(
                    preference_id=str(
                        uuid.uuid4()
                    ),
                    key=key,
                    value=value,
                    category=category,
                    confidence=confidence,
                    source=source,
                    metadata=dict(
                        metadata or {}
                    ),
                )

                self._preferences[
                    preference_key
                ] = preference

        self._persist(
            preference
        )

        return preference.preference_id

    # Aliases commonly used by RENIX components.

    set_preference = set
    save = set
    remember = set

    # ------------------------------------------------------------------
    # Get preference
    # ------------------------------------------------------------------

    def get(
        self,
        key: str,
        *,
        category: str = "general",
        default: Any = None,
        mark_used: bool = True,
    ) -> Any:

        preference_key = self._make_key(
            category,
            key,
        )

        with self._lock:

            preference = (
                self._preferences.get(
                    preference_key
                )
            )

        if preference is None:

            preference = self._load(
                category,
                key,
            )

        if preference is None:
            return default

        if not preference.enabled:
            return default

        if mark_used:

            preference.usage_count += 1

            preference.updated_at = _now()

        return preference.value

    get_preference = get

    # ------------------------------------------------------------------
    # Get complete Preference object
    # ------------------------------------------------------------------

    def get_record(
        self,
        key: str,
        *,
        category: str = "general",
    ) -> Optional[Preference]:

        preference_key = self._make_key(
            category,
            key,
        )

        with self._lock:

            return self._preferences.get(
                preference_key
            )

    # ------------------------------------------------------------------
    # Check preference
    # ------------------------------------------------------------------

    def has(
        self,
        key: str,
        *,
        category: str = "general",
    ) -> bool:

        return (
            self.get_record(
                key,
                category=category,
            )
            is not None
        )

    contains = has

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    def remove(
        self,
        key: str,
        *,
        category: str = "general",
    ) -> bool:

        preference_key = self._make_key(
            category,
            key,
        )

        with self._lock:

            preference = (
                self._preferences.pop(
                    preference_key,
                    None,
                )
            )

        if preference is None:
            return False

        self._delete_persistent(
            preference
        )

        return True

    delete = remove
    forget = remove

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable(
        self,
        key: str,
        *,
        category: str = "general",
    ) -> bool:

        preference = self.get_record(
            key,
            category=category,
        )

        if preference is None:
            return False

        preference.enabled = True

        preference.updated_at = _now()

        self._persist(
            preference
        )

        return True

    def disable(
        self,
        key: str,
        *,
        category: str = "general",
    ) -> bool:

        preference = self.get_record(
            key,
            category=category,
        )

        if preference is None:
            return False

        preference.enabled = False

        preference.updated_at = _now()

        self._persist(
            preference
        )

        return True

    # ------------------------------------------------------------------
    # Category access
    # ------------------------------------------------------------------

    def get_category(
        self,
        category: str,
        *,
        enabled_only: bool = True,
    ) -> Dict[str, Any]:

        category = str(
            category
        ).strip()

        result = {}

        with self._lock:

            for preference in (
                self._preferences.values()
            ):

                if (
                    preference.category
                    != category
                ):
                    continue

                if (
                    enabled_only
                    and not preference.enabled
                ):
                    continue

                result[
                    preference.key
                ] = preference.value

        return result

    # ------------------------------------------------------------------
    # Complete profile
    # ------------------------------------------------------------------

    def get_all(
        self,
        *,
        enabled_only: bool = True,
    ) -> Dict[str, Any]:

        result = {}

        with self._lock:

            for preference in (
                self._preferences.values()
            ):

                if (
                    enabled_only
                    and not preference.enabled
                ):
                    continue

                result[
                    preference.category
                ] = result.get(
                    preference.category,
                    {},
                )

                result[
                    preference.category
                ][
                    preference.key
                ] = preference.value

        return result

    get_profile = get_all
    profile = get_all

    # ------------------------------------------------------------------
    # List records
    # ------------------------------------------------------------------

    def list_preferences(
        self,
        *,
        category: Optional[str] = None,
        enabled_only: bool = True,
    ) -> List[Preference]:

        with self._lock:

            preferences = list(
                self._preferences.values()
            )

        result = []

        for preference in preferences:

            if (
                category is not None
                and preference.category
                != category
            ):
                continue

            if (
                enabled_only
                and not preference.enabled
            ):
                continue

            result.append(
                preference
            )

        result.sort(
            key=lambda item: (
                item.category,
                item.key,
            )
        )

        return result

    # ------------------------------------------------------------------
    # Update confidence
    # ------------------------------------------------------------------

    def update_confidence(
        self,
        key: str,
        confidence: float,
        *,
        category: str = "general",
    ) -> bool:

        preference = self.get_record(
            key,
            category=category,
        )

        if preference is None:
            return False

        preference.confidence = (
            self._clamp(
                confidence
            )
        )

        preference.updated_at = _now()

        self._persist(
            preference
        )

        return True

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------

    def mark_used(
        self,
        key: str,
        *,
        category: str = "general",
    ) -> bool:

        preference = self.get_record(
            key,
            category=category,
        )

        if preference is None:
            return False

        preference.usage_count += 1

        preference.updated_at = _now()

        self._persist(
            preference
        )

        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> List[Preference]:

        query = str(
            query
        ).strip().lower()

        if not query:
            return []

        with self._lock:

            preferences = list(
                self._preferences.values()
            )

        matches = []

        for preference in preferences:

            searchable = " ".join(
                [
                    preference.key,
                    preference.category,
                    str(
                        preference.value
                    ),
                ]
            ).lower()

            if query in searchable:

                matches.append(
                    preference
                )

        matches.sort(
            key=lambda item: (
                item.usage_count,
                item.confidence,
                item.updated_at,
            ),
            reverse=True,
        )

        return matches[
            :max(
                1,
                int(limit),
            )
        ]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(
        self,
    ) -> List[
        Dict[str, Any]
    ]:

        return [
            preference.to_dict()
            for preference
            in self.list_preferences(
                enabled_only=False
            )
        ]

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_preferences(
        self,
        preferences: List[
            Dict[str, Any]
        ],
        *,
        overwrite: bool = False,
    ) -> int:

        imported = 0

        for data in preferences:

            if not isinstance(
                data,
                dict,
            ):
                continue

            key = data.get(
                "key"
            )

            if not key:
                continue

            category = data.get(
                "category",
                "general",
            )

            existing = self.get_record(
                key,
                category=category,
            )

            if (
                existing is not None
                and not overwrite
            ):
                continue

            self.set(
                key,
                data.get("value"),
                category=category,
                confidence=data.get(
                    "confidence",
                    1.0,
                ),
                source=data.get(
                    "source",
                    "import",
                ),
                metadata=data.get(
                    "metadata",
                    {},
                ),
            )

            imported += 1

        return imported

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(
        self,
        preference: Preference,
    ) -> None:

        if self.store is None:
            return

        for method_name in (
            "save_preference",
            "save",
            "store",
            "upsert",
        ):

            method = getattr(
                self.store,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:

                method(
                    preference.to_dict()
                )

                return

            except TypeError:

                try:

                    method(
                        preference
                    )

                    return

                except Exception:
                    continue

            except Exception:
                logger.exception(
                    "Preference persistence failed."
                )
                return

    def _load(
        self,
        category: str,
        key: str,
    ) -> Optional[Preference]:

        if self.store is None:
            return None

        for method_name in (
            "get_preference",
            "get",
            "load",
        ):

            method = getattr(
                self.store,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:

                result = method(
                    self._make_key(
                        category,
                        key,
                    )
                )

                if isinstance(
                    result,
                    Preference,
                ):
                    preference = result

                elif isinstance(
                    result,
                    dict,
                ):

                    preference = Preference(
                        **{
                            field_name: result[
                                field_name
                            ]
                            for field_name
                            in Preference.__dataclass_fields__
                            if field_name in result
                        }
                    )

                else:
                    continue

                with self._lock:

                    self._preferences[
                        self._make_key(
                            category,
                            key,
                        )
                    ] = preference

                return preference

            except Exception:
                continue

        return None

    def _delete_persistent(
        self,
        preference: Preference,
    ) -> None:

        if self.store is None:
            return

        for method_name in (
            "delete_preference",
            "delete",
            "remove",
        ):

            method = getattr(
                self.store,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:

                method(
                    self._make_key(
                        preference.category,
                        preference.key,
                    )
                )

                return

            except Exception:
                continue

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(
        category: str,
        key: str,
    ) -> str:

        return (
            f"{str(category).strip().lower()}"
            f":"
            f"{str(key).strip().lower()}"
        )

    @staticmethod
    def _clamp(
        value: Any,
    ) -> float:

        try:
            value = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            value = 0.0

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )


__all__ = [
    "Preference",
    "PreferenceMemory",
]


