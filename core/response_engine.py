"""
RENIX AI
Core Response Engine

Responsible for:
- Converting internal RENIX results into user-facing responses
- Formatting responses
- Choosing response style
- Handling concise/detailed responses
- Supporting text, voice, and UI output
- Handling errors gracefully
- Maintaining response metadata
- Supporting conversational responses

The Response Engine does NOT execute computer actions.
It presents the result of RENIX's reasoning and execution.
"""

from __future__ import annotations

import logging
import time
import uuid

from dataclasses import dataclass, field
from typing import Any, Optional


logger = logging.getLogger(
    "RENIX.ResponseEngine"
)


# ============================================================================
# RESPONSE
# ============================================================================

@dataclass
class Response:
    """
    Represents a complete RENIX response.
    """

    response_id: str

    text: str

    success: bool = True

    response_type: str = "text"

    tone: str = "natural"

    emotion: str = "neutral"

    confidence: float = 1.0

    should_speak: bool = True

    should_display: bool = True

    should_notify: bool = False

    source: Optional[str] = None

    intent: Optional[str] = None

    action: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: float = field(
        default_factory=time.time
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert the response to a dictionary.
        """

        return {
            "response_id": self.response_id,
            "text": self.text,
            "success": self.success,
            "response_type": self.response_type,
            "tone": self.tone,
            "emotion": self.emotion,
            "confidence": self.confidence,
            "should_speak": self.should_speak,
            "should_display": self.should_display,
            "should_notify": self.should_notify,
            "source": self.source,
            "intent": self.intent,
            "action": self.action,
            "metadata": dict(
                self.metadata
            ),
            "created_at": self.created_at,
        }


# ============================================================================
# RESPONSE ENGINE
# ============================================================================

class ResponseEngine:
    """
    Main RENIX response generation and formatting engine.
    """

    def __init__(
        self,
        personality: Any = None,
        context_engine: Any = None,
        memory_manager: Any = None,
        event_bus: Any = None,
    ) -> None:

        self.logger = logger

        self.personality = (
            personality
        )

        self.context_engine = (
            context_engine
        )

        self.memory_manager = (
            memory_manager
        )

        self.event_bus = event_bus

        self.initialized = False

        self.created_at = time.time()

        self.history: list[
            Response
        ] = []

        self.max_history = 500

        self.default_tone = "natural"

        self.default_emotion = "neutral"

        self.default_confidence = 1.0

        self.default_should_speak = True

        self.default_should_display = True

        self.user_name: Optional[str] = None

        self.conversation_mode = True

        self.verbose_mode = False

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize the response engine.
        """

        self.initialized = True

        self.logger.info(
            "RENIX Response Engine initialized."
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Shut down the response engine.
        """

        self.initialized = False

        self.logger.info(
            "RENIX Response Engine shutdown."
        )

    # ========================================================================
    # ID
    # ========================================================================

    @staticmethod
    def _generate_id(
        prefix: str = "response",
    ) -> str:
        """
        Generate a unique response ID.
        """

        return (
            f"{prefix}_"
            f"{uuid.uuid4().hex}"
        )

    # ========================================================================
    # CLAMP
    # ========================================================================

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        """
        Clamp a number between minimum and maximum.
        """

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            value = minimum

        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )

    # ========================================================================
    # CLEAN TEXT
    # ========================================================================

    @staticmethod
    def clean_text(
        text: Any,
    ) -> str:
        """
        Clean and normalize response text.
        """

        if text is None:

            return ""

        text = str(
            text
        )

        text = text.replace(
            "\x00",
            "",
        )

        text = text.strip()

        return text

    # ========================================================================
    # GET PERSONALITY VALUE
    # ========================================================================

    def _get_personality_value(
        self,
        attribute: str,
        default: Any,
    ) -> Any:
        """
        Safely retrieve a personality setting.
        """

        if self.personality is None:

            return default

        try:

            if hasattr(
                self.personality,
                attribute,
            ):

                value = getattr(
                    self.personality,
                    attribute,
                )

                if callable(value):

                    return value()

                return value

            if hasattr(
                self.personality,
                "get",
            ):

                value = self.personality.get(
                    attribute,
                    default,
                )

                return value

        except Exception as exc:

            self.logger.debug(
                "Unable to read personality setting %s: %s",
                attribute,
                exc,
            )

        return default

    # ========================================================================
    # GET CONTEXT
    # ========================================================================

    def get_context(
        self,
    ) -> dict[str, Any]:
        """
        Retrieve available conversation context.
        """

        if self.context_engine is None:

            return {}

        try:

            if hasattr(
                self.context_engine,
                "get_all",
            ):

                result = (
                    self.context_engine
                    .get_all()
                )

                if isinstance(
                    result,
                    dict,
                ):

                    return result

            if hasattr(
                self.context_engine,
                "get_context",
            ):

                result = (
                    self.context_engine
                    .get_context()
                )

                if isinstance(
                    result,
                    dict,
                ):

                    return result

        except Exception as exc:

            self.logger.debug(
                "Unable to retrieve response context: %s",
                exc,
            )

        return {}

    # ========================================================================
    # EXTRACT TEXT
    # ========================================================================

    def extract_text(
        self,
        value: Any,
    ) -> str:
        """
        Convert many possible internal result types
        into user-facing text.
        """

        if value is None:

            return ""

        if isinstance(
            value,
            str,
        ):

            return self.clean_text(
                value
            )

        if isinstance(
            value,
            bytes,
        ):

            try:

                return self.clean_text(
                    value.decode(
                        "utf-8",
                        errors="replace",
                    )
                )

            except Exception:

                return ""

        if isinstance(
            value,
            dict,
        ):

            preferred_keys = [
                "response",
                "text",
                "message",
                "answer",
                "content",
                "output",
                "result",
                "summary",
                "description",
                "error",
            ]

            for key in preferred_keys:

                if key not in value:

                    continue

                candidate = (
                    value.get(key)
                )

                if candidate is None:

                    continue

                if isinstance(
                    candidate,
                    (str, int, float),
                ):

                    return self.clean_text(
                        candidate
                    )

            # --------------------------------------------------------------
            # Nested result
            # --------------------------------------------------------------

            if "result" in value:

                nested = self.extract_text(
                    value["result"]
                )

                if nested:

                    return nested

            return self.clean_text(
                str(value)
            )

        if isinstance(
            value,
            (list, tuple),
        ):

            if not value:

                return ""

            parts = []

            for item in value:

                text = self.extract_text(
                    item
                )

                if text:

                    parts.append(
                        text
                    )

            return "\n".join(
                parts
            )

        # --------------------------------------------------------------------
        # Object attributes
        # --------------------------------------------------------------------

        for attribute in [
            "text",
            "message",
            "answer",
            "content",
            "output",
            "result",
        ]:

            try:

                if hasattr(
                    value,
                    attribute,
                ):

                    candidate = getattr(
                        value,
                        attribute,
                    )

                    if candidate is value:

                        continue

                    text = self.extract_text(
                        candidate
                    )

                    if text:

                        return text

            except Exception:

                continue

        return self.clean_text(
            value
        )

    # ========================================================================
    # FORMAT RESULT
    # ========================================================================

    def format_result(
        self,
        result: Any,
        *,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> str:
        """
        Format an internal result for the user.
        """

        text = self.extract_text(
            result
        )

        if prefix:

            text = (
                self.clean_text(
                    prefix
                )
                + " "
                + text
            )

        if suffix:

            text = (
                text
                + " "
                + self.clean_text(
                    suffix
                )
            )

        return self.clean_text(
            text
        )

    # ========================================================================
    # SUCCESS RESPONSE
    # ========================================================================

    def success(
        self,
        text: Any,
        *,
        tone: Optional[str] = None,
        emotion: Optional[str] = None,
        confidence: float = 1.0,
        response_type: str = "text",
        should_speak: Optional[bool] = None,
        should_display: Optional[bool] = None,
        should_notify: bool = False,
        source: Optional[str] = None,
        intent: Optional[str] = None,
        action: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Response:
        """
        Create a successful response.
        """

        response = Response(
            response_id=self._generate_id(),
            text=self.clean_text(
                text
            ),
            success=True,
            response_type=response_type,
            tone=(
                tone
                or self.default_tone
            ),
            emotion=(
                emotion
                or self.default_emotion
            ),
            confidence=self._clamp(
                confidence
            ),
            should_speak=(
                self.default_should_speak
                if should_speak is None
                else should_speak
            ),
            should_display=(
                self.default_should_display
                if should_display is None
                else should_display
            ),
            should_notify=should_notify,
            source=source,
            intent=intent,
            action=action,
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
        )

        self._store_response(
            response
        )

        return response

    # ========================================================================
    # ERROR RESPONSE
    # ========================================================================

    def error(
        self,
        error: Any,
        *,
        tone: Optional[str] = None,
        confidence: float = 1.0,
        should_speak: Optional[bool] = None,
        should_display: Optional[bool] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Response:
        """
        Create an error response.
        """

        error_text = self.clean_text(
            error
        )

        if not error_text:

            error_text = (
                "Something went wrong."
            )

        response = Response(
            response_id=self._generate_id(),
            text=error_text,
            success=False,
            response_type="error",
            tone=(
                tone
                or "apologetic"
            ),
            emotion="concerned",
            confidence=self._clamp(
                confidence
            ),
            should_speak=(
                self.default_should_speak
                if should_speak is None
                else should_speak
            ),
            should_display=(
                self.default_should_display
                if should_display is None
                else should_display
            ),
            should_notify=True,
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
        )

        self._store_response(
            response
        )

        return response

    # ========================================================================
    # ASK RESPONSE
    # ========================================================================

    def ask(
        self,
        question: Any,
        *,
        tone: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Response:
        """
        Create a question/clarification response.
        """

        text = self.clean_text(
            question
        )

        response = Response(
            response_id=self._generate_id(),
            text=text,
            success=True,
            response_type="question",
            tone=(
                tone
                or "curious"
            ),
            emotion="curious",
            confidence=1.0,
            should_speak=True,
            should_display=True,
            should_notify=False,
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
        )

        self._store_response(
            response
        )

        return response

    # ========================================================================
    # INFORMATION RESPONSE
    # ========================================================================

    def information(
        self,
        text: Any,
        *,
        source: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Response:
        """
        Create an informational response.
        """

        return self.success(
            text,
            response_type="information",
            source=source,
            confidence=confidence,
            metadata=metadata,
        )

    # ========================================================================
    # COMMAND RESPONSE
    # ========================================================================

    def command_response(
        self,
        text: Any,
        *,
        action: Optional[str] = None,
        success: bool = True,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Response:
        """
        Create a response related to a command/action.
        """

        if success:

            return self.success(
                text,
                response_type="command",
                action=action,
                metadata=metadata,
            )

        return self.error(
            text,
            metadata={
                **(
                    metadata
                    if metadata
                    else {}
                ),
                "action": action,
            },
        )

    # ========================================================================
    # EXECUTION RESPONSE
    # ========================================================================

    def from_execution(
        self,
        execution_result: Any,
    ) -> Response:
        """
        Convert an ExecutionResult into a response.
        """

        if execution_result is None:

            return self.error(
                "I couldn't get an execution result."
            )

        success = bool(
            getattr(
                execution_result,
                "success",
                False,
            )
        )

        result = getattr(
            execution_result,
            "result",
            None,
        )

        error = getattr(
            execution_result,
            "error",
            None,
        )

        action = getattr(
            execution_result,
            "action",
            None,
        )

        duration = getattr(
            execution_result,
            "duration",
            None,
        )

        metadata = {
            "execution_id": getattr(
                execution_result,
                "execution_id",
                None,
            ),
            "decision_id": getattr(
                execution_result,
                "decision_id",
                None,
            ),
            "status": getattr(
                execution_result,
                "status",
                None,
            ),
            "duration": duration,
        }

        if success:

            text = self.extract_text(
                result
            )

            if not text:

                if action:

                    text = (
                        f"Done. "
                        f"I completed "
                        f"'{action}'."
                    )

                else:

                    text = (
                        "Done. "
                        "The task was completed."
                    )

            return self.success(
                text,
                response_type="execution",
                action=action,
                metadata=metadata,
            )

        error_text = (
            self.extract_text(
                error
            )
        )

        if not error_text:

            error_text = (
                "I couldn't complete that task."
            )

        return self.error(
            error_text,
            metadata=metadata,
        )

    # ========================================================================
    # FROM DECISION
    # ========================================================================

    def from_decision(
        self,
        decision: Any,
    ) -> Response:
        """
        Convert a Decision into a user-facing response.
        """

        if decision is None:

            return self.error(
                "I couldn't make a decision."
            )

        status = getattr(
            decision,
            "status",
            "unknown",
        )

        reason = getattr(
            decision,
            "reason",
            "",
        )

        option = getattr(
            decision,
            "selected_option",
            None,
        )

        if status == "awaiting_confirmation":

            if option is not None:

                name = getattr(
                    option,
                    "name",
                    "this action",
                )

                return self.ask(
                    (
                        f"I can {name}. "
                        "Would you like me to proceed?"
                    ),
                    metadata={
                        "decision_id": getattr(
                            decision,
                            "decision_id",
                            None,
                        ),
                    },
                )

            return self.ask(
                "I need your confirmation before proceeding."
            )

        if status == "approved":

            if option is not None:

                name = getattr(
                    option,
                    "name",
                    "the selected action",
                )

                return self.success(
                    f"Ready to {name}.",
                    response_type="decision",
                )

            return self.success(
                "The decision is approved.",
                response_type="decision",
            )

        if status == "rejected":

            return self.error(
                reason
                or "I couldn't find a suitable action.",
                metadata={
                    "decision_id": getattr(
                        decision,
                        "decision_id",
                        None,
                    ),
                },
            )

        if status == "completed":

            return self.success(
                reason
                or "The decision was completed.",
                response_type="decision",
            )

        return self.information(
            reason
            or f"Decision status: {status}",
            response_type="decision",
        )

    # ========================================================================
    # NATURALIZE
    # ========================================================================

    def naturalize(
        self,
        text: str,
        *,
        context: Optional[
            dict[str, Any]
        ] = None,
    ) -> str:
        """
        Make response text more natural without changing
        its core meaning.
        """

        text = self.clean_text(
            text
        )

        if not text:

            return text

        context = (
            context
            if context is not None
            else self.get_context()
        )

        # --------------------------------------------------------------
        # Remove unnecessary technical punctuation
        # --------------------------------------------------------------

        text = text.replace(
            " .",
            ".",
        )

        text = text.replace(
            " ,",
            ",",
        )

        # --------------------------------------------------------------
        # User name
        # --------------------------------------------------------------

        user_name = (
            self.user_name
            or context.get(
                "user_name"
            )
            if isinstance(
                context,
                dict,
            )
            else None
        )

        # Avoid repeatedly inserting names.
        if (
            user_name
            and len(text) < 120
            and text.lower().startswith(
                "hello"
            )
        ):

            text = (
                f"Hello {user_name}. "
                + text[
                    6:
                ].strip()
            )

        return text

    # ========================================================================
    # PERSONALIZE
    # ========================================================================

    def personalize(
        self,
        text: str,
        *,
        context: Optional[
            dict[str, Any]
        ] = None,
    ) -> str:
        """
        Apply lightweight personalization.
        """

        text = self.clean_text(
            text
        )

        if not text:

            return text

        context = (
            context
            if context is not None
            else self.get_context()
        )

        user_name = (
            self.user_name
            if self.user_name
            else (
                context.get(
                    "user_name"
                )
                if isinstance(
                    context,
                    dict,
                )
                else None
            )
        )

        # Do not automatically insert the user's name
        # into every response. Only replace explicit
        # placeholders.
        if user_name:

            text = text.replace(
                "{user_name}",
                str(
                    user_name
                ),
            )

        return text

    # ========================================================================
    # BUILD RESPONSE
    # ========================================================================

    def build(
        self,
        text: Any,
        *,
        success: bool = True,
        tone: Optional[str] = None,
        emotion: Optional[str] = None,
        confidence: float = 1.0,
        response_type: str = "text",
        should_speak: Optional[bool] = None,
        should_display: Optional[bool] = None,
        should_notify: bool = False,
        source: Optional[str] = None,
        intent: Optional[str] = None,
        action: Optional[str] = None,
        personalize: bool = True,
        naturalize: bool = True,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Response:
        """
        Build a complete response.
        """

        text_value = self.extract_text(
            text
        )

        context = self.get_context()

        if personalize:

            text_value = self.personalize(
                text_value,
                context=context,
            )

        if naturalize:

            text_value = self.naturalize(
                text_value,
                context=context,
            )

        if success:

            return self.success(
                text_value,
                tone=tone,
                emotion=emotion,
                confidence=confidence,
                response_type=response_type,
                should_speak=should_speak,
                should_display=should_display,
                should_notify=should_notify,
                source=source,
                intent=intent,
                action=action,
                metadata=metadata,
            )

        return self.error(
            text_value,
            tone=tone,
            confidence=confidence,
            should_speak=should_speak,
            should_display=should_display,
            metadata=metadata,
        )

    # ========================================================================
    # STORE RESPONSE
    # ========================================================================

    def _store_response(
        self,
        response: Response,
    ) -> None:
        """
        Store response in local history.
        """

        self.history.append(
            response
        )

        if len(
            self.history
        ) > self.max_history:

            self.history = self.history[
                -self.max_history:
            ]

        self._save_to_memory(
            response
        )

    # ========================================================================
    # SAVE TO MEMORY
    # ========================================================================

    def _save_to_memory(
        self,
        response: Response,
    ) -> None:
        """
        Save the response to RENIX memory when
        a compatible memory manager exists.
        """

        if self.memory_manager is None:

            return

        try:

            if hasattr(
                self.memory_manager,
                "add_conversation",
            ):

                self.memory_manager.add_conversation(
                    role="assistant",
                    content=response.text,
                    metadata=response.to_dict(),
                )

                return

            if hasattr(
                self.memory_manager,
                "remember",
            ):

                self.memory_manager.remember(
                    response.text,
                    metadata=response.to_dict(),
                )

        except Exception as exc:

            self.logger.debug(
                "Unable to save response to memory: %s",
                exc,
            )

    # ========================================================================
    # GET HISTORY
    # ========================================================================

    def get_history(
        self,
        limit: Optional[int] = None,
    ) -> list[Response]:
        """
        Return response history.
        """

        if limit is None:

            limit = self.max_history

        limit = max(
            0,
            int(limit),
        )

        if limit == 0:

            return []

        return self.history[
            -limit:
        ]

    # ========================================================================
    # LAST RESPONSE
    # ========================================================================

    def last_response(
        self,
    ) -> Optional[Response]:
        """
        Return the most recent response.
        """

        if not self.history:

            return None

        return self.history[
            -1
        ]

    # ========================================================================
    # CLEAR HISTORY
    # ========================================================================

    def clear_history(
        self,
    ) -> None:
        """
        Clear response history.
        """

        self.history.clear()

    # ========================================================================
    # SET USER NAME
    # ========================================================================

    def set_user_name(
        self,
        name: Optional[str],
    ) -> None:
        """
        Set the user's preferred name.
        """

        if name is None:

            self.user_name = None

            return

        name = self.clean_text(
            name
        )

        self.user_name = (
            name
            if name
            else None
        )

    # ========================================================================
    # SET TONE
    # ========================================================================

    def set_tone(
        self,
        tone: str,
    ) -> None:
        """
        Set the default response tone.
        """

        tone = self.clean_text(
            tone
        )

        if tone:

            self.default_tone = tone

    # ========================================================================
    # SET EMOTION
    # ========================================================================

    def set_emotion(
        self,
        emotion: str,
    ) -> None:
        """
        Set the default response emotion.
        """

        emotion = self.clean_text(
            emotion
        )

        if emotion:

            self.default_emotion = (
                emotion
            )

    # ========================================================================
    # SET VERBOSE MODE
    # ========================================================================

    def set_verbose_mode(
        self,
        enabled: bool,
    ) -> None:
        """
        Enable or disable verbose responses.
        """

        self.verbose_mode = bool(
            enabled
        )

    # ========================================================================
    # SET CONVERSATION MODE
    # ========================================================================

    def set_conversation_mode(
        self,
        enabled: bool,
    ) -> None:
        """
        Enable or disable conversational behavior.
        """

        self.conversation_mode = bool(
            enabled
        )

    # ========================================================================
    # FORMAT LIST
    # ========================================================================

    def format_list(
        self,
        items: list[Any],
        *,
        numbered: bool = False,
    ) -> str:
        """
        Convert a list into readable response text.
        """

        if not items:

            return ""

        lines = []

        for index, item in enumerate(
            items,
            start=1,
        ):

            text = self.extract_text(
                item
            )

            if not text:

                continue

            if numbered:

                lines.append(
                    f"{index}. {text}"
                )

            else:

                lines.append(
                    f"• {text}"
                )

        return "\n".join(
            lines
        )

    # ========================================================================
    # FORMAT TABLE
    # ========================================================================

    def format_table(
        self,
        rows: list[
            dict[str, Any]
        ],
    ) -> str:
        """
        Convert dictionaries into a simple text table.
        """

        if not rows:

            return ""

        columns = []

        for row in rows:

            for key in row.keys():

                if key not in columns:

                    columns.append(
                        str(key)
                    )

        if not columns:

            return ""

        widths = {
            column: len(
                column
            )
            for column in columns
        }

        for row in rows:

            for column in columns:

                value = self.clean_text(
                    row.get(
                        column,
                        "",
                    )
                )

                widths[column] = max(
                    widths[column],
                    len(value),
                )

        header = " | ".join(
            column.ljust(
                widths[column]
            )
            for column in columns
        )

        separator = "-+-".join(
            "-" * widths[column]
            for column in columns
        )

        lines = [
            header,
            separator,
        ]

        for row in rows:

            lines.append(
                " | ".join(
                    self.clean_text(
                        row.get(
                            column,
                            "",
                        )
                    ).ljust(
                        widths[column]
                    )
                    for column in columns
                )
            )

        return "\n".join(
            lines
        )

    # ========================================================================
    # EVENT
    # ========================================================================

    async def emit_response_event(
        self,
        response: Response,
    ) -> None:
        """
        Emit a response-created event.
        """

        if self.event_bus is None:

            return

        payload = response.to_dict()

        try:

            if hasattr(
                self.event_bus,
                "publish",
            ):

                result = (
                    self.event_bus.publish(
                        "response.created",
                        payload,
                    )
                )

                if hasattr(
                    result,
                    "__await__",
                ):

                    await result

                return

            if hasattr(
                self.event_bus,
                "emit",
            ):

                result = (
                    self.event_bus.emit(
                        "response.created",
                        payload,
                    )
                )

                if hasattr(
                    result,
                    "__await__",
                ):

                    await result

        except Exception as exc:

            self.logger.debug(
                "Unable to emit response event: %s",
                exc,
            )

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return response engine statistics.
        """

        return {
            "initialized": (
                self.initialized
            ),
            "history_size": len(
                self.history
            ),
            "max_history": (
                self.max_history
            ),
            "default_tone": (
                self.default_tone
            ),
            "default_emotion": (
                self.default_emotion
            ),
            "conversation_mode": (
                self.conversation_mode
            ),
            "verbose_mode": (
                self.verbose_mode
            ),
            "user_name_set": (
                self.user_name is not None
            ),
            "created_at": (
                self.created_at
            ),
        }


# ============================================================================
# GLOBAL RESPONSE ENGINE
# ============================================================================

response_engine = ResponseEngine()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def respond(
    text: Any,
    *,
    success: bool = True,
    response_type: str = "text",
) -> Response:
    """
    Convenience wrapper for creating a response.
    """

    return response_engine.build(
        text,
        success=success,
        response_type=response_type,
    )


def respond_error(
    error: Any,
) -> Response:
    """
    Convenience wrapper for error responses.
    """

    return response_engine.error(
        error
    )


def respond_success(
    text: Any,
) -> Response:
    """
    Convenience wrapper for successful responses.
    """

    return response_engine.success(
        text
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "Response",
    "ResponseEngine",
    "response_engine",
    "respond",
    "respond_error",
    "respond_success",
]


