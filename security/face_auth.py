"""
RENIX Face Authentication
=========================

Provides face-based authentication for the RENIX AI system.

Features:

* Face authentication
* User face registration
* Face verification
* Confidence threshold
* Multiple registered users
* Authentication cooldown
* Integration with RENIX vision system
  """

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("RENIX.FaceAuth")

class FaceAuth:
    """
    Face authentication system for RENIX.

    This module acts as a security layer over
    the vision/face recognition system.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        face_recognition_engine: Optional[Any] = None,
    ) -> None:
        """
        Initialize face authentication.

        Args:
            config:
                Face authentication configuration.

            face_recognition_engine:
                Optional RENIX face recognition engine.
        """

        self.config = config or {}

        self.enabled = self.config.get(
            "enabled",
            True,
        )

        self.confidence_threshold = float(
            self.config.get(
                "confidence_threshold",
                0.85,
            )
        )

        self.authentication_cooldown = float(
            self.config.get(
                "authentication_cooldown",
                2.0,
            )
        )

        self.max_failed_attempts = int(
            self.config.get(
                "max_failed_attempts",
                5,
            )
        )

        self.face_recognition_engine = (
            face_recognition_engine
        )

        self.registered_users: Dict[
            str,
            Dict[str, Any]
        ] = {}

        self.failed_attempts: Dict[
            str,
            int
        ] = {}

        self.last_authentication_time = 0.0

        self.last_authenticated_user: Optional[
            str
        ] = None

        logger.info(
            "FaceAuth initialized."
        )

    # ============================================================
    # ENGINE MANAGEMENT
    # ============================================================

    def set_face_recognition_engine(
        self,
        engine: Any,
    ) -> None:
        """
        Set the face recognition engine.
        """

        self.face_recognition_engine = engine

        logger.info(
            "Face recognition engine connected."
        )

    def is_engine_available(
        self,
    ) -> bool:
        """
        Check whether a face recognition engine
        is available.
        """

        return (
            self.face_recognition_engine
            is not None
        )

    # ============================================================
    # USER REGISTRATION
    # ============================================================

    def register_face(
        self,
        user_id: str,
        face_data: Any,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> bool:
        """
        Register a face for a user.

        Args:
            user_id:
                Unique RENIX user identifier.

            face_data:
                Face image, encoding, or camera data.

            metadata:
                Optional additional information.

        Returns:
            True if registration succeeds.
        """

        if not self.enabled:
            return False

        if not user_id:
            logger.warning(
                "Face registration requires user_id."
            )
            return False

        try:

            face_encoding = (
                self._create_face_encoding(
                    face_data
                )
            )

            if face_encoding is None:

                logger.warning(
                    "Unable to create face encoding."
                )

                return False

            self.registered_users[user_id] = {
                "face_encoding": face_encoding,
                "registered_at": (
                    datetime.now().isoformat()
                ),
                "metadata": metadata or {},
                "enabled": True,
            }

            self.failed_attempts[
                user_id
            ] = 0

            logger.info(
                "Face registered for user: %s",
                user_id,
            )

            return True

        except Exception as error:

            logger.error(
                "Face registration failed: %s",
                error,
            )

            return False

    # ============================================================
    # REMOVE FACE
    # ============================================================

    def remove_face(
        self,
        user_id: str,
    ) -> bool:
        """
        Remove a registered face.
        """

        if user_id not in self.registered_users:
            return False

        del self.registered_users[user_id]

        self.failed_attempts.pop(
            user_id,
            None,
        )

        logger.info(
            "Face removed for user: %s",
            user_id,
        )

        return True

    # ============================================================
    # ENABLE / DISABLE USER
    # ============================================================

    def enable_user(
        self,
        user_id: str,
    ) -> bool:
        """
        Enable face authentication for a user.
        """

        user = self.registered_users.get(
            user_id
        )

        if user is None:
            return False

        user["enabled"] = True

        return True

    def disable_user(
        self,
        user_id: str,
    ) -> bool:
        """
        Disable face authentication for a user.
        """

        user = self.registered_users.get(
            user_id
        )

        if user is None:
            return False

        user["enabled"] = False

        return True

    # ============================================================
    # FACE ENCODING
    # ============================================================

    def _create_face_encoding(
        self,
        face_data: Any,
    ) -> Optional[Any]:
        """
        Create a face encoding.

        If RENIX's face recognition engine is
        available, it is used.
        """

        if (
            self.face_recognition_engine
            is None
        ):

            # Store raw data as fallback.
            # A real engine should be connected
            # for production face recognition.
            return face_data

        try:

            engine = (
                self.face_recognition_engine
            )

            if hasattr(
                engine,
                "encode_face",
            ):

                return engine.encode_face(
                    face_data
                )

            if hasattr(
                engine,
                "get_face_encoding",
            ):

                return engine.get_face_encoding(
                    face_data
                )

            if hasattr(
                engine,
                "extract_embedding",
            ):

                return engine.extract_embedding(
                    face_data
                )

            logger.warning(
                "Face engine has no supported "
                "encoding method."
            )

            return None

        except Exception as error:

            logger.error(
                "Face encoding failed: %s",
                error,
            )

            return None

    # ============================================================
    # FACE COMPARISON
    # ============================================================

    def _compare_faces(
        self,
        known_face: Any,
        current_face: Any,
    ) -> float:
        """
        Compare two face encodings.

        Returns:
            Confidence score between 0 and 1.
        """

        if (
            self.face_recognition_engine
            is not None
        ):

            try:

                engine = (
                    self.face_recognition_engine
                )

                if hasattr(
                    engine,
                    "compare_faces",
                ):

                    result = engine.compare_faces(
                        known_face,
                        current_face,
                    )

                    if isinstance(
                        result,
                        bool,
                    ):

                        return (
                            1.0
                            if result
                            else 0.0
                        )

                    return float(result)

                if hasattr(
                    engine,
                    "compare",
                ):

                    result = engine.compare(
                        known_face,
                        current_face,
                    )

                    return float(result)

                if hasattr(
                    engine,
                    "calculate_similarity",
                ):

                    return float(
                        engine.calculate_similarity(
                            known_face,
                            current_face,
                        )
                    )

            except Exception as error:

                logger.error(
                    "Face comparison failed: %s",
                    error,
                )

                return 0.0

        # Basic fallback comparison.
        if known_face == current_face:
            return 1.0

        return 0.0

    # ============================================================
    # AUTHENTICATE USER
    # ============================================================

    def authenticate(
        self,
        user_id: str,
        face_data: Any,
    ) -> bool:
        """
        Authenticate a user using face data.

        Args:
            user_id:
                Expected user identity.

            face_data:
                Current camera face data.

        Returns:
            True if authentication succeeds.
        """

        result = self.verify(
            user_id=user_id,
            face_data=face_data,
        )

        return bool(
            result.get(
                "authenticated",
                False,
            )
        )

    # ============================================================
    # VERIFY FACE
    # ============================================================

    def verify(
        self,
        user_id: str,
        face_data: Any,
    ) -> Dict[str, Any]:
        """
        Verify a face against a registered user.

        Returns detailed authentication result.
        """

        result = {
            "authenticated": False,
            "user_id": user_id,
            "confidence": 0.0,
            "reason": "",
        }

        if not self.enabled:

            result["reason"] = (
                "Face authentication disabled."
            )

            return result

        # Cooldown protection
        current_time = time.time()

        if (
            current_time
            - self.last_authentication_time
            < self.authentication_cooldown
        ):

            result["reason"] = (
                "Authentication cooldown active."
            )

            return result

        user = self.registered_users.get(
            user_id
        )

        if user is None:

            result["reason"] = (
                "User face not registered."
            )

            return result

        if not user.get(
            "enabled",
            False,
        ):

            result["reason"] = (
                "Face authentication disabled "
                "for this user."
            )

            return result

        # Too many failed attempts
        failed_count = (
            self.failed_attempts.get(
                user_id,
                0,
            )
        )

        if (
            failed_count
            >= self.max_failed_attempts
        ):

            result["reason"] = (
                "Maximum authentication attempts "
                "reached."
            )

            return result

        try:

            current_encoding = (
                self._create_face_encoding(
                    face_data
                )
            )

            if current_encoding is None:

                result["reason"] = (
                    "Unable to detect face."
                )

                return result

            confidence = (
                self._compare_faces(
                    user[
                        "face_encoding"
                    ],
                    current_encoding,
                )
            )

            result[
                "confidence"
            ] = confidence

            self.last_authentication_time = (
                current_time
            )

            if (
                confidence
                >= self.confidence_threshold
            ):

                result[
                    "authenticated"
                ] = True

                result[
                    "reason"
                ] = (
                    "Face authentication successful."
                )

                self.failed_attempts[
                    user_id
                ] = 0

                self.last_authenticated_user = (
                    user_id
                )

                logger.info(
                    "Face authentication successful: %s",
                    user_id,
                )

            else:

                result[
                    "reason"
                ] = (
                    "Face verification confidence "
                    "below threshold."
                )

                self.failed_attempts[
                    user_id
                ] = (
                    self.failed_attempts.get(
                        user_id,
                        0,
                    )
                    + 1
                )

                logger.warning(
                    "Face authentication failed: %s",
                    user_id,
                )

            return result

        except Exception as error:

            logger.error(
                "Face authentication error: %s",
                error,
            )

            result["reason"] = str(
                error
            )

            return result

    # ============================================================
    # IDENTIFY USER
    # ============================================================

    def identify(
        self,
        face_data: Any,
    ) -> Dict[str, Any]:
        """
        Identify which registered user matches
        the provided face.

        Returns:
            Best matching user and confidence.
        """

        result = {
            "identified": False,
            "user_id": None,
            "confidence": 0.0,
        }

        if not self.enabled:
            return result

        try:

            current_encoding = (
                self._create_face_encoding(
                    face_data
                )
            )

            if current_encoding is None:
                return result

            best_user = None
            best_confidence = 0.0

            for (
                user_id,
                user,
            ) in self.registered_users.items():

                if not user.get(
                    "enabled",
                    False,
                ):
                    continue

                confidence = (
                    self._compare_faces(
                        user[
                            "face_encoding"
                        ],
                        current_encoding,
                    )
                )

                if (
                    confidence
                    > best_confidence
                ):

                    best_confidence = (
                        confidence
                    )

                    best_user = user_id

            result[
                "confidence"
            ] = best_confidence

            if (
                best_user is not None
                and best_confidence
                >= self.confidence_threshold
            ):

                result[
                    "identified"
                ] = True

                result[
                    "user_id"
                ] = best_user

                self.last_authenticated_user = (
                    best_user
                )

            return result

        except Exception as error:

            logger.error(
                "Face identification failed: %s",
                error,
            )

            return result

    # ============================================================
    # RESET FAILED ATTEMPTS
    # ============================================================

    def reset_failed_attempts(
        self,
        user_id: str,
    ) -> None:
        """
        Reset authentication failure counter.
        """

        self.failed_attempts[
            user_id
        ] = 0

    # ============================================================
    # USER INFORMATION
    # ============================================================

    def get_registered_users(
        self,
    ) -> List[str]:
        """
        Return registered user IDs.
        """

        return list(
            self.registered_users.keys()
        )

    def is_user_registered(
        self,
        user_id: str,
    ) -> bool:
        """
        Check whether a user has a
        registered face.
        """

        return (
            user_id
            in self.registered_users
        )

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return face authentication status.
        """

        enabled_users = sum(
            1
            for user
            in self.registered_users.values()
            if user.get(
                "enabled",
                False,
            )
        )

        return {
            "enabled": self.enabled,
            "engine_available": (
                self.is_engine_available()
            ),
            "registered_users": len(
                self.registered_users
            ),
            "enabled_users": enabled_users,
            "confidence_threshold": (
                self.confidence_threshold
            ),
            "last_authenticated_user": (
                self.last_authenticated_user
            ),
        }

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Safely shut down face authentication.
        """

        logger.info(
            "Shutting down FaceAuth..."
        )

        self.last_authenticated_user = None
        self.failed_attempts.clear()

        logger.info(
            "FaceAuth shutdown complete."
        )
