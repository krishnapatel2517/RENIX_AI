"""
RENIX Authentication System
===========================

Provides authentication management for the RENIX AI system.

Supported authentication methods:

* Password
* PIN
* Token
* Session authentication
* Custom authentication handlers
  """

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger("RENIX.Authentication")

class AuthenticationManager:
    """
    Authentication manager for RENIX.

    Handles user authentication and session tokens.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize authentication manager.
        """

        self.config = config or {}

        self.enabled = self.config.get(
            "enabled",
            True,
        )

        self.session_duration = self.config.get(
            "session_duration",
            3600,
        )

        self.users: Dict[
            str,
            Dict[str, Any]
        ] = {}

        self.sessions: Dict[
            str,
            Dict[str, Any]
        ] = {}

        self._initialize_default_user()

        logger.info(
            "AuthenticationManager initialized."
        )

    # ============================================================
    # USER INITIALIZATION
    # ============================================================

    def _initialize_default_user(
        self,
    ) -> None:
        """
        Create the default RENIX user if configured.
        """

        username = self.config.get(
            "default_username",
            "default",
        )

        password = self.config.get(
            "default_password",
            None,
        )

        pin = self.config.get(
            "default_pin",
            None,
        )

        if password or pin:

            self.register_user(
                username=username,
                password=password,
                pin=pin,
            )

    # ============================================================
    # PASSWORD HASHING
    # ============================================================

    @staticmethod
    def _hash_value(
        value: str,
        salt: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Hash a password or PIN using PBKDF2.
        """

        if salt is None:
            salt = secrets.token_hex(16)

        hashed = hashlib.pbkdf2_hmac(
            "sha256",
            value.encode("utf-8"),
            salt.encode("utf-8"),
            100_000,
        )

        return {
            "salt": salt,
            "hash": hashed.hex(),
        }

    # ============================================================
    # PASSWORD VERIFICATION
    # ============================================================

    @staticmethod
    def _verify_value(
        value: str,
        stored_hash: str,
        salt: str,
    ) -> bool:
        """
        Verify a password or PIN.
        """

        new_hash = hashlib.pbkdf2_hmac(
            "sha256",
            value.encode("utf-8"),
            salt.encode("utf-8"),
            100_000,
        ).hex()

        return hmac.compare_digest(
            new_hash,
            stored_hash,
        )

    # ============================================================
    # REGISTER USER
    # ============================================================

    def register_user(
        self,
        username: str,
        password: Optional[str] = None,
        pin: Optional[str] = None,
    ) -> bool:
        """
        Register a new user.

        At least one authentication method
        must be provided.
        """

        if not username:
            return False

        if not password and not pin:
            logger.warning(
                "User registration requires password or PIN."
            )
            return False

        if username in self.users:
            logger.warning(
                "User already exists: %s",
                username,
            )
            return False

        user_data: Dict[str, Any] = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "password": None,
            "pin": None,
            "enabled": True,
        }

        if password:

            user_data["password"] = (
                self._hash_value(password)
            )

        if pin:

            user_data["pin"] = (
                self._hash_value(pin)
            )

        self.users[username] = user_data

        logger.info(
            "User registered: %s",
            username,
        )

        return True

    # ============================================================
    # REMOVE USER
    # ============================================================

    def remove_user(
        self,
        username: str,
    ) -> bool:
        """
        Remove a user.
        """

        if username not in self.users:
            return False

        del self.users[username]

        self._remove_user_sessions(
            username
        )

        logger.info(
            "User removed: %s",
            username,
        )

        return True

    # ============================================================
    # ENABLE USER
    # ============================================================

    def enable_user(
        self,
        username: str,
    ) -> bool:
        """
        Enable a user account.
        """

        user = self.users.get(username)

        if user is None:
            return False

        user["enabled"] = True

        return True

    # ============================================================
    # DISABLE USER
    # ============================================================

    def disable_user(
        self,
        username: str,
    ) -> bool:
        """
        Disable a user account.
        """

        user = self.users.get(username)

        if user is None:
            return False

        user["enabled"] = False

        self._remove_user_sessions(
            username
        )

        logger.warning(
            "User disabled: %s",
            username,
        )

        return True

    # ============================================================
    # AUTHENTICATE USER
    # ============================================================

    def authenticate(
        self,
        credentials: Dict[str, Any],
    ) -> bool:
        """
        Authenticate a user.

        Credentials may contain:
        - username
        - password
        - pin
        - token
        """

        if not self.enabled:
            return True

        username = credentials.get(
            "username",
            "default",
        )

        # Token authentication
        token = credentials.get(
            "token"
        )

        if token:

            return self.validate_session(
                token
            )

        user = self.users.get(
            username
        )

        if user is None:

            logger.warning(
                "Authentication failed. Unknown user: %s",
                username,
            )

            return False

        if not user.get(
            "enabled",
            False,
        ):

            logger.warning(
                "Authentication failed. User disabled: %s",
                username,
            )

            return False

        # Password authentication
        password = credentials.get(
            "password"
        )

        if password and user.get(
            "password"
        ):

            password_data = user[
                "password"
            ]

            if self._verify_value(
                password,
                password_data["hash"],
                password_data["salt"],
            ):

                logger.info(
                    "Password authentication successful: %s",
                    username,
                )

                return True

        # PIN authentication
        pin = credentials.get(
            "pin"
        )

        if pin and user.get(
            "pin"
        ):

            pin_data = user["pin"]

            if self._verify_value(
                pin,
                pin_data["hash"],
                pin_data["salt"],
            ):

                logger.info(
                    "PIN authentication successful: %s",
                    username,
                )

                return True

        logger.warning(
            "Authentication failed: %s",
            username,
        )

        return False

    # ============================================================
    # LOGIN
    # ============================================================

    def login(
        self,
        credentials: Dict[str, Any],
    ) -> Optional[str]:
        """
        Authenticate a user and create a session.

        Returns:
            Session token if successful.
        """

        if not self.authenticate(
            credentials
        ):

            return None

        username = credentials.get(
            "username",
            "default",
        )

        token = self.create_session(
            username
        )

        logger.info(
            "User logged in: %s",
            username,
        )

        return token

    # ============================================================
    # CREATE SESSION
    # ============================================================

    def create_session(
        self,
        username: str,
    ) -> str:
        """
        Create an authenticated session.
        """

        token = secrets.token_urlsafe(
            32
        )

        expires_at = (
            datetime.now()
            + timedelta(
                seconds=self.session_duration
            )
        )

        self.sessions[token] = {
            "username": username,
            "created_at": datetime.now(),
            "expires_at": expires_at,
            "active": True,
        }

        return token

    # ============================================================
    # VALIDATE SESSION
    # ============================================================

    def validate_session(
        self,
        token: str,
    ) -> bool:
        """
        Validate an authentication session.
        """

        session = self.sessions.get(
            token
        )

        if session is None:
            return False

        if not session.get(
            "active",
            False,
        ):

            return False

        expires_at = session.get(
            "expires_at"
        )

        if expires_at and datetime.now() > expires_at:

            session["active"] = False

            return False

        return True

    # ============================================================
    # GET SESSION USER
    # ============================================================

    def get_session_user(
        self,
        token: str,
    ) -> Optional[str]:
        """
        Return the username associated with a session.
        """

        if not self.validate_session(
            token
        ):

            return None

        session = self.sessions.get(
            token
        )

        if session is None:
            return None

        return session.get(
            "username"
        )

    # ============================================================
    # LOGOUT
    # ============================================================

    def logout(
        self,
        token: str,
    ) -> bool:
        """
        End an authentication session.
        """

        session = self.sessions.get(
            token
        )

        if session is None:
            return False

        session["active"] = False

        logger.info(
            "Session logged out."
        )

        return True

    # ============================================================
    # REMOVE USER SESSIONS
    # ============================================================

    def _remove_user_sessions(
        self,
        username: str,
    ) -> None:
        """
        Invalidate all sessions belonging to a user.
        """

        for token, session in (
            self.sessions.items()
        ):

            if session.get(
                "username"
            ) == username:

                session["active"] = False

    # ============================================================
    # CLEAN EXPIRED SESSIONS
    # ============================================================

    def cleanup_sessions(
        self,
    ) -> int:
        """
        Remove expired sessions.

        Returns:
            Number of removed sessions.
        """

        expired_tokens = []

        for token, session in (
            self.sessions.items()
        ):

            expires_at = session.get(
                "expires_at"
            )

            if (
                expires_at
                and datetime.now() > expires_at
            ):

                expired_tokens.append(
                    token
                )

        for token in expired_tokens:

            del self.sessions[token]

        if expired_tokens:

            logger.info(
                "Removed %d expired sessions.",
                len(expired_tokens),
            )

        return len(expired_tokens)

    # ============================================================
    # CHANGE PASSWORD
    # ============================================================

    def change_password(
        self,
        username: str,
        old_password: str,
        new_password: str,
    ) -> bool:
        """
        Change a user's password.
        """

        user = self.users.get(
            username
        )

        if user is None:
            return False

        password_data = user.get(
            "password"
        )

        if password_data is None:
            return False

        if not self._verify_value(
            old_password,
            password_data["hash"],
            password_data["salt"],
        ):

            logger.warning(
                "Password change failed for user: %s",
                username,
            )

            return False

        user["password"] = self._hash_value(
            new_password
        )

        self._remove_user_sessions(
            username
        )

        logger.info(
            "Password changed for user: %s",
            username,
        )

        return True

    # ============================================================
    # CHANGE PIN
    # ============================================================

    def change_pin(
        self,
        username: str,
        old_pin: str,
        new_pin: str,
    ) -> bool:
        """
        Change a user's PIN.
        """

        user = self.users.get(
            username
        )

        if user is None:
            return False

        pin_data = user.get(
            "pin"
        )

        if pin_data is None:
            return False

        if not self._verify_value(
            old_pin,
            pin_data["hash"],
            pin_data["salt"],
        ):

            logger.warning(
                "PIN change failed for user: %s",
                username,
            )

            return False

        user["pin"] = self._hash_value(
            new_pin
        )

        self._remove_user_sessions(
            username
        )

        logger.info(
            "PIN changed for user: %s",
            username,
        )

        return True

    # ============================================================
    # USER STATUS
    # ============================================================

    def user_exists(
        self,
        username: str,
    ) -> bool:
        """
        Check whether a user exists.
        """

        return username in self.users

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return authentication system status.
        """

        active_sessions = sum(
            1
            for session in self.sessions.values()
            if session.get(
                "active",
                False,
            )
        )

        return {
            "enabled": self.enabled,
            "registered_users": len(
                self.users
            ),
            "active_sessions": active_sessions,
            "total_sessions": len(
                self.sessions
            ),
        }

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Safely shut down authentication manager.
        """

        logger.info(
            "Shutting down AuthenticationManager..."
        )

        for session in self.sessions.values():

            session["active"] = False

        self.sessions.clear()

        logger.info(
            "AuthenticationManager shutdown complete."
        )
