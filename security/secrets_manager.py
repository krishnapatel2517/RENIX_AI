"""
RENIX Secrets Manager
=====================

Centralized secret management for RENIX.

This module manages sensitive configuration values such as:

* API keys
* Access tokens
* Passwords
* Service credentials
* Private configuration values

Features:

* In-memory secret storage
* Environment variable loading
* Optional encrypted persistent storage
* Secret masking
* Secret rotation
* Expiration support
* Access tracking
* Secret metadata
* Secure deletion
* Event callbacks

Important:
This module is designed as a security layer inside RENIX.
Production deployments should use a dedicated secret-management
service or OS credential vault where appropriate.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("RENIX.SecretsManager")

class SecretsManager:
    """
    Centralized secret manager for RENIX.

    Example:

        secrets_manager = SecretsManager()

        secrets_manager.set_secret(
            name="OPENAI_API_KEY",
            value="example-secret"
        )

        api_key = secrets_manager.get_secret(
            "OPENAI_API_KEY"
        )
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize SecretsManager.
        """

        self.config = config or {}

        self.secrets: Dict[
            str,
            Dict[str, Any],
        ] = {}

        self.event_handlers: List[
            Callable[
                [Dict[str, Any]],
                None,
            ]
        ] = []

        self.history: List[
            Dict[str, Any]
        ] = []

        self.max_history = int(
            self.config.get(
                "max_history",
                1000,
            )
        )

        self.auto_load_environment = bool(
            self.config.get(
                "auto_load_environment",
                False,
            )
        )

        self.environment_prefix = str(
            self.config.get(
                "environment_prefix",
                "",
            )
        )

        self.storage_path = Path(
            self.config.get(
                "storage_path",
                "data/security/secrets.enc",
            )
        )

        self.enable_persistence = bool(
            self.config.get(
                "enable_persistence",
                False,
            )
        )

        self.master_key = (
            self.config.get(
                "master_key"
            )
            or os.getenv(
                "RENIX_MASTER_KEY"
            )
        )

        self._lock = threading.RLock()

        if self.auto_load_environment:

            self.load_environment_secrets()

        if self.enable_persistence:

            self.load_from_storage()

        logger.info(
            "SecretsManager initialized."
        )

    # ============================================================
    # SECRET NAME VALIDATION
    # ============================================================

    @staticmethod
    def normalize_name(
        name: str,
    ) -> str:
        """
        Normalize secret name.
        """

        if not isinstance(
            name,
            str,
        ):
            raise TypeError(
                "Secret name must be a string."
            )

        name = (
            name.strip()
            .upper()
            .replace(
                " ",
                "_",
            )
        )

        if not name:

            raise ValueError(
                "Secret name cannot be empty."
            )

        return name

    # ============================================================
    # SET SECRET
    # ============================================================

    def set_secret(
        self,
        name: str,
        value: str,
        expires_in: Optional[
            int
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        source: str = "manual",
        overwrite: bool = True,
    ) -> bool:
        """
        Store a secret.

        Args:
            name:
                Unique secret name.

            value:
                Secret value.

            expires_in:
                Optional expiration time in seconds.

            metadata:
                Optional metadata.

            source:
                Source of the secret.

            overwrite:
                Whether existing secrets can be replaced.
        """

        name = self.normalize_name(
            name
        )

        if value is None:

            raise ValueError(
                "Secret value cannot be None."
            )

        value = str(value)

        with self._lock:

            if (
                name in self.secrets
                and not overwrite
            ):

                return False

            now = datetime.now()

            expires_at = None

            if expires_in is not None:

                expires_at = (
                    now
                    + timedelta(
                        seconds=max(
                            1,
                            int(
                                expires_in
                            ),
                        )
                    )
                )

            secret_data = {
                "value": value,
                "created_at": (
                    now.isoformat()
                ),
                "updated_at": (
                    now.isoformat()
                ),
                "expires_at": (
                    expires_at.isoformat()
                    if expires_at
                    else None
                ),
                "metadata": (
                    dict(metadata)
                    if metadata
                    else {}
                ),
                "source": source,
                "access_count": 0,
                "last_accessed": None,
            }

            existed = (
                name in self.secrets
            )

            self.secrets[
                name
            ] = secret_data

            event = {
                "event": (
                    "secret_updated"
                    if existed
                    else "secret_created"
                ),
                "name": name,
                "source": source,
                "timestamp": (
                    datetime.now().isoformat()
                ),
            }

            self._add_history(
                event
            )

        logger.info(
            "Secret stored: %s",
            name,
        )

        self._notify(
            event
        )

        if self.enable_persistence:

            self.save_to_storage()

        return True

    # ============================================================
    # GET SECRET
    # ============================================================

    def get_secret(
        self,
        name: str,
        default: Optional[
            str
        ] = None,
        track_access: bool = True,
    ) -> Optional[str]:
        """
        Retrieve a secret value.
        """

        name = self.normalize_name(
            name
        )

        with self._lock:

            self.cleanup_expired()

            secret_data = (
                self.secrets.get(
                    name
                )
            )

            if secret_data is None:

                return default

            if track_access:

                secret_data[
                    "access_count"
                ] += 1

                secret_data[
                    "last_accessed"
                ] = (
                    datetime.now().isoformat()
                )

                event = {
                    "event": (
                        "secret_accessed"
                    ),
                    "name": name,
                    "timestamp": (
                        datetime.now().isoformat()
                    ),
                }

                self._add_history(
                    event
                )

            return secret_data[
                "value"
            ]

    # ============================================================
    # SECRET EXISTENCE
    # ============================================================

    def has_secret(
        self,
        name: str,
    ) -> bool:
        """
        Check whether a secret exists.
        """

        name = self.normalize_name(
            name
        )

        with self._lock:

            self.cleanup_expired()

            return (
                name
                in self.secrets
            )

    # ============================================================
    # DELETE SECRET
    # ============================================================

    def delete_secret(
        self,
        name: str,
    ) -> bool:
        """
        Remove a secret.
        """

        name = self.normalize_name(
            name
        )

        with self._lock:

            if (
                name
                not in self.secrets
            ):

                return False

            del self.secrets[
                name
            ]

            event = {
                "event": (
                    "secret_deleted"
                ),
                "name": name,
                "timestamp": (
                    datetime.now().isoformat()
                ),
            }

            self._add_history(
                event
            )

        logger.warning(
            "Secret deleted: %s",
            name,
        )

        self._notify(
            event
        )

        if self.enable_persistence:

            self.save_to_storage()

        return True

    # ============================================================
    # ROTATE SECRET
    # ============================================================

    def rotate_secret(
        self,
        name: str,
        new_value: str,
        expires_in: Optional[
            int
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> bool:
        """
        Replace an existing secret with a new value.
        """

        name = self.normalize_name(
            name
        )

        with self._lock:

            if (
                name
                not in self.secrets
            ):

                return False

            old_secret = (
                self.secrets[
                    name
                ]
            )

            source = (
                old_secret.get(
                    "source",
                    "rotation",
                )
            )

        result = self.set_secret(
            name=name,
            value=new_value,
            expires_in=expires_in,
            metadata=metadata,
            source=source,
            overwrite=True,
        )

        if result:

            event = {
                "event": (
                    "secret_rotated"
                ),
                "name": name,
                "timestamp": (
                    datetime.now().isoformat()
                ),
            }

            with self._lock:

                self._add_history(
                    event
                )

            self._notify(
                event
            )

        return result

    # ============================================================
    # ENVIRONMENT VARIABLES
    # ============================================================

    def load_environment_secret(
        self,
        name: str,
        env_name: Optional[
            str
        ] = None,
        required: bool = False,
    ) -> bool:
        """
        Load a secret from an environment variable.
        """

        name = self.normalize_name(
            name
        )

        env_name = (
            env_name
            or name
        )

        value = os.getenv(
            env_name
        )

        if value is None:

            if required:

                raise RuntimeError(
                    f"Required environment secret "
                    f"'{env_name}' was not found."
                )

            return False

        return self.set_secret(
            name=name,
            value=value,
            source=(
                f"environment:{env_name}"
            ),
        )

    def load_environment_secrets(
        self,
        prefix: Optional[
            str
        ] = None,
    ) -> int:
        """
        Load environment variables as secrets.

        If prefix is supplied, only variables
        beginning with that prefix are loaded.
        """

        prefix = (
            prefix
            if prefix is not None
            else self.environment_prefix
        )

        count = 0

        for (
            key,
            value,
        ) in os.environ.items():

            if (
                prefix
                and not key.startswith(
                    prefix
                )
            ):

                continue

            secret_name = key

            if prefix:

                secret_name = (
                    key[
                        len(prefix):
                    ]
                )

            if not secret_name:

                continue

            try:

                self.set_secret(
                    name=secret_name,
                    value=value,
                    source=(
                        f"environment:{key}"
                    ),
                )

                count += 1

            except Exception as error:

                logger.error(
                    "Failed to load environment "
                    "secret %s: %s",
                    key,
                    error,
                )

        logger.info(
            "Loaded %s environment secrets.",
            count,
        )

        return count

    # ============================================================
    # EXPIRATION
    # ============================================================

    def cleanup_expired(
        self,
    ) -> int:
        """
        Remove expired secrets.
        """

        now = datetime.now()

        expired_names = []

        with self._lock:

            for (
                name,
                secret_data,
            ) in list(
                self.secrets.items()
            ):

                expires_at = (
                    secret_data.get(
                        "expires_at"
                    )
                )

                if not expires_at:

                    continue

                try:

                    expiry = (
                        datetime.fromisoformat(
                            expires_at
                        )
                    )

                    if now >= expiry:

                        expired_names.append(
                            name
                        )

                except Exception:

                    logger.warning(
                        "Invalid expiration for "
                        "secret: %s",
                        name,
                    )

            for name in expired_names:

                del self.secrets[
                    name
                ]

                event = {
                    "event": (
                        "secret_expired"
                    ),
                    "name": name,
                    "timestamp": (
                        datetime.now().isoformat()
                    ),
                }

                self._add_history(
                    event
                )

        for name in expired_names:

            logger.info(
                "Expired secret removed: %s",
                name,
            )

        return len(
            expired_names
        )

    # ============================================================
    # METADATA
    # ============================================================

    def get_metadata(
        self,
        name: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Return secret metadata without exposing
        the secret value.
        """

        name = self.normalize_name(
            name
        )

        with self._lock:

            self.cleanup_expired()

            secret_data = (
                self.secrets.get(
                    name
                )
            )

            if secret_data is None:

                return None

            return {
                "name": name,
                "created_at": (
                    secret_data.get(
                        "created_at"
                    )
                ),
                "updated_at": (
                    secret_data.get(
                        "updated_at"
                    )
                ),
                "expires_at": (
                    secret_data.get(
                        "expires_at"
                    )
                ),
                "metadata": dict(
                    secret_data.get(
                        "metadata",
                        {},
                    )
                ),
                "source": (
                    secret_data.get(
                        "source"
                    )
                ),
                "access_count": (
                    secret_data.get(
                        "access_count",
                        0,
                    )
                ),
                "last_accessed": (
                    secret_data.get(
                        "last_accessed"
                    )
                ),
                "masked_value": (
                    self.mask_secret(
                        secret_data.get(
                            "value",
                            ""
                        )
                    )
                ),
            }

    def list_secrets(
        self,
    ) -> List[str]:
        """
        List secret names.

        Secret values are never returned.
        """

        with self._lock:

            self.cleanup_expired()

            return sorted(
                self.secrets.keys()
            )

    # ============================================================
    # SECRET MASKING
    # ============================================================

    @staticmethod
    def mask_secret(
        value: str,
        visible_start: int = 4,
        visible_end: int = 2,
    ) -> str:
        """
        Mask a secret for safe display.
        """

        if value is None:

            return ""

        value = str(value)

        if len(value) <= 4:

            return "*" * len(value)

        visible_start = max(
            0,
            visible_start,
        )

        visible_end = max(
            0,
            visible_end,
        )

        if (
            visible_start
            + visible_end
            >= len(value)
        ):

            return (
                value[:1]
                + "*"
                * max(
                    1,
                    len(value) - 2,
                )
                + value[-1:]
            )

        hidden_length = (
            len(value)
            - visible_start
            - visible_end
        )

        return (
            value[:visible_start]
            + ("*" * hidden_length)
            + (
                value[-visible_end:]
                if visible_end
                else ""
            )
        )

    # ============================================================
    # SIMPLE ENCRYPTION STORAGE
    # ============================================================

    def _derive_key(
        self,
    ) -> bytes:
        """
        Derive internal storage key.

        This is lightweight obfuscation and should
        not replace dedicated cryptographic key
        management in production.
        """

        key = self.master_key

        if not key:

            key = (
                "RENIX_DEFAULT_LOCAL_KEY"
            )

            logger.warning(
                "No RENIX_MASTER_KEY configured. "
                "Persistent secret storage is using "
                "a default local key."
            )

        return hashlib.sha256(
            key.encode(
                "utf-8"
            )
        ).digest()

    def _xor_cipher(
        self,
        data: bytes,
        key: bytes,
    ) -> bytes:
        """
        Internal reversible byte transformation.

        Intended only for local lightweight storage
        protection. Production deployments should use
        a dedicated cryptographic secret store.
        """

        output = bytearray()

        for index, byte in enumerate(
            data
        ):

            output.append(
                byte
                ^ key[
                    index
                    % len(key)
                ]
            )

        return bytes(
            output
        )

    def _serialize_secrets(
        self,
    ) -> bytes:
        """
        Serialize secrets.
        """

        safe_data = json.dumps(
            self.secrets,
            ensure_ascii=False,
            indent=2,
        )

        return safe_data.encode(
            "utf-8"
        )

    # ============================================================
    # PERSISTENCE
    # ============================================================

    def save_to_storage(
        self,
    ) -> bool:
        """
        Save secrets to local protected storage.
        """

        if not self.enable_persistence:

            return False

        try:

            with self._lock:

                self.cleanup_expired()

                raw_data = (
                    self._serialize_secrets()
                )

                key = (
                    self._derive_key()
                )

                encrypted_data = (
                    self._xor_cipher(
                        raw_data,
                        key,
                    )
                )

                encoded_data = (
                    base64.b64encode(
                        encrypted_data
                    )
                )

                self.storage_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                temporary_path = (
                    self.storage_path.with_suffix(
                        ".tmp"
                    )
                )

                with open(
                    temporary_path,
                    "wb",
                ) as file:

                    file.write(
                        encoded_data
                    )

                temporary_path.replace(
                    self.storage_path
                )

            logger.info(
                "Secrets saved to protected storage."
            )

            return True

        except Exception as error:

            logger.error(
                "Failed to save secrets: %s",
                error,
            )

            return False

    def load_from_storage(
        self,
    ) -> bool:
        """
        Load secrets from local protected storage.
        """

        if not self.storage_path.exists():

            return False

        try:

            with open(
                self.storage_path,
                "rb",
            ) as file:

                encoded_data = (
                    file.read()
                )

            encrypted_data = (
                base64.b64decode(
                    encoded_data
                )
            )

            key = self._derive_key()

            raw_data = (
                self._xor_cipher(
                    encrypted_data,
                    key,
                )
            )

            loaded_data = json.loads(
                raw_data.decode(
                    "utf-8"
                )
            )

            if not isinstance(
                loaded_data,
                dict,
            ):

                raise ValueError(
                    "Invalid secret storage format."
                )

            with self._lock:

                self.secrets.update(
                    loaded_data
                )

                self.cleanup_expired()

            logger.info(
                "Secrets loaded from storage."
            )

            return True

        except Exception as error:

            logger.error(
                "Failed to load secrets: %s",
                error,
            )

            return False

    # ============================================================
    # RANDOM SECRET GENERATION
    # ============================================================

    @staticmethod
    def generate_secret(
        length: int = 32,
    ) -> str:
        """
        Generate a cryptographically secure
        random secret.
        """

        length = max(
            16,
            int(length),
        )

        token = (
            secrets.token_urlsafe(
                length
            )
        )

        return token

    # ============================================================
    # EVENT SYSTEM
    # ============================================================

    def add_event_handler(
        self,
        handler: Callable[
            [Dict[str, Any]],
            None,
        ],
    ) -> None:
        """
        Register event handler.
        """

        if not callable(
            handler
        ):

            return

        with self._lock:

            if (
                handler
                not in self.event_handlers
            ):

                self.event_handlers.append(
                    handler
                )

    def remove_event_handler(
        self,
        handler: Callable,
    ) -> bool:
        """
        Remove event handler.
        """

        with self._lock:

            if (
                handler
                not in self.event_handlers
            ):

                return False

            self.event_handlers.remove(
                handler
            )

            return True

    def _notify(
        self,
        event: Dict[str, Any],
    ) -> None:
        """
        Notify event handlers.
        """

        handlers = list(
            self.event_handlers
        )

        for handler in handlers:

            try:

                handler(
                    event.copy()
                )

            except Exception as error:

                logger.error(
                    "SecretsManager event handler "
                    "failed: %s",
                    error,
                )

    # ============================================================
    # HISTORY
    # ============================================================

    def _add_history(
        self,
        event: Dict[str, Any],
    ) -> None:
        """
        Add event to history.
        """

        self.history.append(
            event
        )

        if (
            len(self.history)
            > self.max_history
        ):

            overflow = (
                len(self.history)
                - self.max_history
            )

            self.history = (
                self.history[
                    overflow:
                ]
            )

    def get_history(
        self,
        limit: int = 100,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Return secret operation history.

        Secret values are never included.
        """

        with self._lock:

            return [
                event.copy()
                for event in self.history[
                    -max(
                        1,
                        limit,
                    ):
                ]
            ]

    def clear_history(
        self,
    ) -> None:
        """
        Clear operation history.
        """

        with self._lock:

            self.history.clear()

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return SecretsManager status.
        """

        with self._lock:

            self.cleanup_expired()

            return {
                "total_secrets": (
                    len(self.secrets)
                ),
                "secret_names": (
                    self.list_secrets()
                ),
                "persistence_enabled": (
                    self.enable_persistence
                ),
                "storage_path": str(
                    self.storage_path
                ),
                "environment_prefix": (
                    self.environment_prefix
                ),
                "history_entries": (
                    len(self.history)
                ),
            }

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Shutdown SecretsManager.
        """

        logger.info(
            "Shutting down SecretsManager..."
        )

        if self.enable_persistence:

            self.save_to_storage()

        with self._lock:

            self.event_handlers.clear()

        logger.info(
            "SecretsManager shutdown complete."
        )
