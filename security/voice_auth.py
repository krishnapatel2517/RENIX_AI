"""
RENIX Voice Authentication
==========================

Provides voice-based authentication for the RENIX AI system.

Features:

* Voice enrollment
* Voice verification
* Speaker authentication
* Confidence threshold
* Failed attempt tracking
* Authentication cooldown
* Integration with RENIX speaker identification system
  """

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("RENIX.VoiceAuth")

class VoiceAuth:
    """
    Voice authentication system for RENIX.

    This module connects with the voice processing
    and speaker identification systems to verify
    a user's identity through voice characteristics.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        speaker_engine: Optional[Any] = None,
    ) -> None:
        """
        Initialize voice authentication.
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

        self.speaker_engine = speaker_engine

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
            "VoiceAuth initialized."
        )

    # ============================================================
    # ENGINE MANAGEMENT
    # ============================================================

    def set_speaker_engine(
        self,
        engine: Any,
    ) -> None:
        """
        Connect a speaker identification engine.
        """

        self.speaker_engine = engine

        logger.info(
            "Speaker identification engine connected."
        )

    def is_engine_available(
        self,
    ) -> bool:
        """
        Check whether a speaker engine is available.
        """

        return self.speaker_engine is not None

    # ============================================================
    # VOICE ENROLLMENT
    # ============================================================

    def register_voice(
        self,
        user_id: str,
        voice_data: Any,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> bool:
        """
        Register voice data for a user.

        Args:
            user_id:
                Unique user identifier.

            voice_data:
                Audio sample or voice embedding.

            metadata:
                Optional user metadata.
        """

        if not self.enabled:
            return False

        if not user_id:

            logger.warning(
                "Voice registration requires user_id."
            )

            return False

        try:

            voice_embedding = (
                self._create_voice_embedding(
                    voice_data
                )
            )

            if voice_embedding is None:

                logger.warning(
                    "Unable to create voice embedding."
                )

                return False

            self.registered_users[user_id] = {
                "voice_embedding": voice_embedding,
                "registered_at": (
                    datetime.now().isoformat()
                ),
                "metadata": metadata or {},
                "enabled": True,
            }

            self.failed_attempts[user_id] = 0

            logger.info(
                "Voice registered for user: %s",
                user_id,
            )

            return True

        except Exception as error:

            logger.error(
                "Voice registration failed: %s",
                error,
            )

            return False

    # ============================================================
    # REMOVE VOICE
    # ============================================================

    def remove_voice(
        self,
        user_id: str,
    ) -> bool:
        """
        Remove a registered voice.
        """

        if user_id not in self.registered_users:
            return False

        del self.registered_users[user_id]

        self.failed_attempts.pop(
            user_id,
            None,
        )

        logger.info(
            "Voice removed for user: %s",
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
        Enable voice authentication for a user.
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
        Disable voice authentication for a user.
        """

        user = self.registered_users.get(
            user_id
        )

        if user is None:
            return False

        user["enabled"] = False

        return True

    # ============================================================
    # CREATE VOICE EMBEDDING
    # ============================================================

    def _create_voice_embedding(
        self,
        voice_data: Any,
    ) -> Optional[Any]:
        """
        Create a voice embedding.

        Uses the connected speaker engine when
        available.
        """

        if self.speaker_engine is None:

            # Development fallback.
            # Production should use a proper
            # speaker embedding model.
            return voice_data

        try:

            engine = self.speaker_engine

            if hasattr(
                engine,
                "extract_embedding",
            ):

                return engine.extract_embedding(
                    voice_data
                )

            if hasattr(
                engine,
                "get_voice_embedding",
            ):

                return engine.get_voice_embedding(
                    voice_data
                )

            if hasattr(
                engine,
                "encode_voice",
            ):

                return engine.encode_voice(
                    voice_data
                )

            logger.warning(
                "Speaker engine has no supported "
                "embedding method."
            )

            return None

        except Exception as error:

            logger.error(
                "Voice embedding creation failed: %s",
                error,
            )

            return None

    # ============================================================
    # COMPARE VOICES
    # ============================================================

    def _compare_voices(
        self,
        registered_voice: Any,
        current_voice: Any,
    ) -> float:
        """
        Compare two voice embeddings.

        Returns a confidence score between 0 and 1.
        """

        if self.speaker_engine is not None:

            try:

                engine = self.speaker_engine

                if hasattr(
                    engine,
                    "compare_voices",
                ):

                    result = engine.compare_voices(
                        registered_voice,
                        current_voice,
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

                    return float(
                        engine.compare(
                            registered_voice,
                            current_voice,
                        )
                    )

                if hasattr(
                    engine,
                    "calculate_similarity",
                ):

                    return float(
                        engine.calculate_similarity(
                            registered_voice,
                            current_voice,
                        )
                    )

            except Exception as error:

                logger.error(
                    "Voice comparison failed: %s",
                    error,
                )

                return 0.0

        # Development fallback comparison.
        if registered_voice == current_voice:
            return 1.0

        return 0.0

    # ============================================================
    # AUTHENTICATE USER
    # ============================================================

    def authenticate(
        self,
        user_id: str,
        voice_data: Any,
    ) -> bool:
        """
        Authenticate a user using voice data.
        """

        result = self.verify(
            user_id=user_id,
            voice_data=voice_data,
        )

        return bool(
            result.get(
                "authenticated",
                False,
            )
        )

    # ============================================================
    # VERIFY VOICE
    # ============================================================

    def verify(
        self,
        user_id: str,
        voice_data: Any,
    ) -> Dict[str, Any]:
        """
        Verify voice data against a registered user.
        """

        result = {
            "authenticated": False,
            "user_id": user_id,
            "confidence": 0.0,
            "reason": "",
        }

        if not self.enabled:

            result["reason"] = (
                "Voice authentication is disabled."
            )

            return result

        current_time = time.time()

        # Authentication cooldown.
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
                "User voice is not registered."
            )

            return result

        if not user.get(
            "enabled",
            False,
        ):

            result["reason"] = (
                "Voice authentication disabled "
                "for this user."
            )

            return result

        failed_count = self.failed_attempts.get(
            user_id,
            0,
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

            current_embedding = (
                self._create_voice_embedding(
                    voice_data
                )
            )

            if current_embedding is None:

                result["reason"] = (
                    "Unable to process voice."
                )

                return result

            confidence = self._compare_voices(
                user["voice_embedding"],
                current_embedding,
            )

            result["confidence"] = confidence

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

                result["reason"] = (
                    "Voice authentication successful."
                )

                self.failed_attempts[
                    user_id
                ] = 0

                self.last_authenticated_user = (
                    user_id
                )

                logger.info(
                    "Voice authentication successful: %s",
                    user_id,
                )

            else:

                result["reason"] = (
                    "Voice confidence below threshold."
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
                    "Voice authentication failed: %s",
                    user_id,
                )

            return result

        except Exception as error:

            logger.error(
                "Voice authentication error: %s",
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
        voice_data: Any,
    ) -> Dict[str, Any]:
        """
        Identify a registered user from voice data.
        """

        result = {
            "identified": False,
            "user_id": None,
            "confidence": 0.0,
        }

        if not self.enabled:
            return result

        try:

            current_embedding = (
                self._create_voice_embedding(
                    voice_data
                )
            )

            if current_embedding is None:
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

                confidence = self._compare_voices(
                    user["voice_embedding"],
                    current_embedding,
                )

                if confidence > best_confidence:

                    best_confidence = confidence
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
                "Voice identification failed: %s",
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
        Reset failed authentication attempts.
        """

        self.failed_attempts[
            user_id
        ] = 0

    # ============================================================
    # REGISTERED USERS
    # ============================================================

    def get_registered_users(
        self,
    ) -> List[str]:
        """
        Return all registered voice users.
        """

        return list(
            self.registered_users.keys()
        )

    def is_user_registered(
        self,
        user_id: str,
    ) -> bool:
        """
        Check whether a user has registered voice data.
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
        Return voice authentication status.
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
        Safely shut down voice authentication.
        """

        logger.info(
            "Shutting down VoiceAuth..."
        )

        self.last_authenticated_user = None
        self.failed_attempts.clear()

        logger.info(
            "VoiceAuth shutdown complete."
        )
