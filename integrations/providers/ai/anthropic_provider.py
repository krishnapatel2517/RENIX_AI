"""
RENIX Anthropic Provider
========================

Anthropic/Claude implementation of RENIX's AIProvider
interface.

Expected environment variable:

    ANTHROPIC_API_KEY

Optional:

    RENIX_ANTHROPIC_MODEL
    ANTHROPIC_BASE_URL

The Anthropic SDK is imported lazily so RENIX can start even
when the Anthropic package is not installed.
"""

from __future__ import annotations

import os
import time
import uuid

from typing import Any, Iterator

from .provider import (
    AIProvider,
    AIRequest,
    AIResponse,
    AIStreamChunk,
    ProviderAuthenticationError,
    ProviderCapabilities,
    ProviderConfig,
    ProviderConnectionError,
    ProviderNotSupportedError,
    ProviderResponseError,
    ProviderStatus,
    ProviderTimeoutError,
    ResponseType,
)


# ============================================================
# ANTHROPIC PROVIDER
# ============================================================


class AnthropicProvider(AIProvider):
    """
    RENIX provider implementation for Anthropic Claude.
    """

    def __init__(
        self,
        config: ProviderConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:

        if config is None:
            config = ProviderConfig(
                name="anthropic",
                api_key=os.getenv(
                    "ANTHROPIC_API_KEY"
                ),
                base_url=os.getenv(
                    "ANTHROPIC_BASE_URL"
                ),
                default_model=os.getenv(
                    "RENIX_ANTHROPIC_MODEL",
                    "claude-sonnet-4-5",
                ),
            )

        capabilities = ProviderCapabilities(
            chat=True,
            streaming=True,
            vision=True,
            audio_input=False,
            audio_output=False,
            embeddings=False,
            tool_calling=True,
            structured_output=True,
            reasoning=True,
            supported_models=[],
        )

        super().__init__(
            config,
            capabilities,
        )

        self._client = client

    # ========================================================
    # PROVIDER NAME
    # ========================================================

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""

        return "anthropic"

    # ========================================================
    # CLIENT
    # ========================================================

    def _create_client(self) -> Any:
        """Create the Anthropic SDK client lazily."""

        try:
            import anthropic
        except ImportError as exc:
            raise ProviderConnectionError(
                "The Anthropic Python SDK is not installed. "
                "Install the 'anthropic' package."
            ) from exc

        api_key = self.config.api_key

        if not api_key:
            raise ProviderAuthenticationError(
                "Anthropic API key is not configured. "
                "Set ANTHROPIC_API_KEY or provide it through "
                "RENIX's secrets manager."
            )

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }

        if self.config.base_url:
            kwargs["base_url"] = (
                self.config.base_url
            )

        try:
            return anthropic.Anthropic(
                **kwargs
            )
        except Exception as exc:
            raise ProviderConnectionError(
                f"Failed to initialize Anthropic client: "
                f"{exc}"
            ) from exc

    @property
    def client(self) -> Any:
        """Return lazily initialized Anthropic client."""

        if self._client is None:
            self._client = self._create_client()

        return self._client

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def initialize(self) -> None:
        """Initialize Anthropic provider."""

        if not self.enabled:
            self.status = (
                ProviderStatus.DISABLED
            )
            return

        self._initialized = True

        if self.config.api_key:
            self.status = (
                ProviderStatus.UNKNOWN
            )
        else:
            self.status = (
                ProviderStatus.UNAVAILABLE
            )

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def health_check(self) -> bool:
        """
        Perform a lightweight Anthropic availability check.

        Anthropic does not require a generation request merely
        to initialize the provider. Credential presence is
        therefore used as the lightweight local check.
        """

        if not self.enabled:
            self.status = (
                ProviderStatus.DISABLED
            )
            return False

        if not self.config.api_key:
            self.status = (
                ProviderStatus.UNAVAILABLE
            )
            return False

        try:
            # Constructing the client validates local SDK
            # configuration without consuming model tokens.
            self.client

            self.status = (
                ProviderStatus.AVAILABLE
            )

            return True

        except Exception:
            self.status = (
                ProviderStatus.ERROR
            )
            return False

    # ========================================================
    # REQUEST HELPERS
    # ========================================================

    def _build_messages(
        self,
        request: AIRequest,
    ) -> list[dict[str, Any]]:
        """
        Convert a RENIX request into Anthropic message format.
        """

        return [
            {
                "role": "user",
                "content": request.prompt,
            }
        ]

    def _build_tools(
        self,
        request: AIRequest,
    ) -> list[dict[str, Any]] | None:
        """Convert RENIX tool definitions."""

        if not request.tools:
            return None

        tools: list[
            dict[str, Any]
        ] = []

        for tool in request.tools:
            normalized = dict(tool)

            # RENIX may already provide Anthropic-compatible
            # tool schemas. Preserve those unchanged.
            tools.append(normalized)

        return tools

    def _build_request_kwargs(
        self,
        request: AIRequest,
    ) -> dict[str, Any]:
        """Build Anthropic Messages API parameters."""

        model = self.resolve_model(
            request.model
        )

        if not model:
            raise ProviderResponseError(
                "No Anthropic model was specified."
            )

        max_tokens = (
            request.max_tokens
            or int(
                self.config.extra.get(
                    "default_max_tokens",
                    2048,
                )
            )
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._build_messages(
                request
            ),
            "max_tokens": max_tokens,
        }

        if request.system_prompt:
            kwargs["system"] = (
                request.system_prompt
            )

        if request.temperature is not None:
            kwargs["temperature"] = (
                request.temperature
            )

        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        if request.stop_sequences:
            kwargs["stop_sequences"] = list(
                request.stop_sequences
            )

        tools = self._build_tools(
            request
        )

        if tools:
            kwargs["tools"] = tools

        # Anthropic's Messages API returns JSON/text according
        # to the model/API behavior. ResponseType.JSON is
        # therefore retained in RENIX metadata rather than
        # sending an unsupported generic response_format field.
        return kwargs

    # ========================================================
    # RESPONSE HELPERS
    # ========================================================

    @staticmethod
    def _extract_text(
        response: Any,
    ) -> str:
        """Extract text blocks from an Anthropic response."""

        content = getattr(
            response,
            "content",
            None,
        )

        if not content:
            return ""

        parts: list[str] = []

        for block in content:
            block_type = getattr(
                block,
                "type",
                None,
            )

            if block_type == "text":
                text = getattr(
                    block,
                    "text",
                    None,
                )

                if text:
                    parts.append(
                        str(text)
                    )

        return "".join(parts)

    @staticmethod
    def _extract_tool_calls(
        response: Any,
    ) -> list[dict[str, Any]]:
        """Extract Anthropic tool-use blocks."""

        content = getattr(
            response,
            "content",
            None,
        )

        if not content:
            return []

        result: list[
            dict[str, Any]
        ] = []

        for block in content:
            block_type = getattr(
                block,
                "type",
                None,
            )

            if block_type != "tool_use":
                continue

            result.append(
                {
                    "id": getattr(
                        block,
                        "id",
                        None,
                    ),
                    "type": "function",
                    "name": getattr(
                        block,
                        "name",
                        None,
                    ),
                    "arguments": getattr(
                        block,
                        "input",
                        None,
                    ),
                }
            )

        return result

    @staticmethod
    def _extract_usage(
        response: Any,
    ) -> tuple[int, int, int]:
        """Extract Anthropic token usage."""

        usage = getattr(
            response,
            "usage",
            None,
        )

        if usage is None:
            return 0, 0, 0

        input_tokens = int(
            getattr(
                usage,
                "input_tokens",
                0,
            )
            or 0
        )

        output_tokens = int(
            getattr(
                usage,
                "output_tokens",
                0,
            )
            or 0
        )

        total_tokens = (
            input_tokens
            + output_tokens
        )

        return (
            input_tokens,
            output_tokens,
            total_tokens,
        )

    @staticmethod
    def _extract_request_id(
        response: Any,
    ) -> str | None:
        """Extract Anthropic response ID."""

        value = getattr(
            response,
            "id",
            None,
        )

        if value:
            return str(value)

        return None

    @staticmethod
    def _extract_finish_reason(
        response: Any,
    ) -> str | None:
        """Extract Anthropic stop reason."""

        value = getattr(
            response,
            "stop_reason",
            None,
        )

        if value:
            return str(value)

        return None

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        request: AIRequest,
    ) -> AIResponse:
        """Generate a complete Claude response."""

        self.validate_request(
            request
        )

        self.record_request()

        started = time.perf_counter()

        try:
            response = (
                self.client.messages.create(
                    **self._build_request_kwargs(
                        request
                    )
                )
            )

            latency_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            (
                input_tokens,
                output_tokens,
                total_tokens,
            ) = self._extract_usage(
                response
            )

            result = AIResponse(
                text=self._extract_text(
                    response
                ),
                model=self.resolve_model(
                    request.model
                ),
                provider=self.provider_name,
                request_id=(
                    self._extract_request_id(
                        response
                    )
                    or uuid.uuid4().hex
                ),
                finish_reason=(
                    self._extract_finish_reason(
                        response
                    )
                ),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                tool_calls=(
                    self._extract_tool_calls(
                        response
                    )
                ),
                raw_response=response,
                metadata={
                    "provider": "anthropic",
                },
            )

            self.status = (
                ProviderStatus.AVAILABLE
            )

            self.record_response(
                result
            )

            return result

        except TimeoutError as exc:
            self.record_error(exc)

            raise ProviderTimeoutError(
                f"Anthropic request timed out: "
                f"{exc}"
            ) from exc

        except Exception as exc:
            self.record_error(exc)

            raise self._translate_error(
                exc
            ) from exc

    # ========================================================
    # STREAM
    # ========================================================

    def stream(
        self,
        request: AIRequest,
    ) -> Iterator[AIStreamChunk]:
        """Stream a Claude response."""

        self.validate_request(
            request
        )

        self.record_request()

        self.metrics.streamed_requests += 1

        started = time.perf_counter()

        try:
            kwargs = self._build_request_kwargs(
                request
            )

            request_id: str | None = None

            # The Anthropic SDK's streaming context manager
            # exposes server-sent events.
            with self.client.messages.stream(
                **kwargs
            ) as stream:

                for index, event in enumerate(
                    stream
                ):
                    event_type = getattr(
                        event,
                        "type",
                        None,
                    )

                    text = (
                        self._extract_stream_text(
                            event
                        )
                    )

                    if event_type == "message_start":
                        message = getattr(
                            event,
                            "message",
                            None,
                        )

                        request_id = (
                            getattr(
                                message,
                                "id",
                                None,
                            )
                            or request_id
                        )

                    finish_reason = (
                        self._extract_stream_finish_reason(
                            event
                        )
                    )

                    is_final = (
                        event_type
                        == "message_stop"
                        or finish_reason
                        is not None
                    )

                    yield AIStreamChunk(
                        text=text,
                        provider=self.provider_name,
                        model=self.resolve_model(
                            request.model
                        ),
                        request_id=request_id,
                        index=index,
                        is_final=is_final,
                        finish_reason=finish_reason,
                        metadata={
                            "provider": "anthropic",
                            "event_type": event_type,
                        },
                    )

            latency_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            self.status = (
                ProviderStatus.AVAILABLE
            )

            self.metrics.last_success_at = (
                time.time()
            )

            self.metrics.total_latency_ms += (
                latency_ms
            )

        except TimeoutError as exc:
            self.record_error(exc)

            raise ProviderTimeoutError(
                f"Anthropic streaming request timed out: "
                f"{exc}"
            ) from exc

        except Exception as exc:
            self.record_error(exc)

            raise self._translate_error(
                exc
            ) from exc

    # ========================================================
    # STREAM EVENT PARSING
    # ========================================================

    @staticmethod
    def _extract_stream_text(
        event: Any,
    ) -> str:
        """
        Extract text from Anthropic stream events.
        """

        event_type = getattr(
            event,
            "type",
            None,
        )

        # text_delta events expose:
        #
        # event.delta.text
        #
        if event_type == "content_block_delta":
            delta = getattr(
                event,
                "delta",
                None,
            )

            if delta is None:
                return ""

            delta_type = getattr(
                delta,
                "type",
                None,
            )

            if delta_type == "text_delta":
                text = getattr(
                    delta,
                    "text",
                    None,
                )

                return (
                    str(text)
                    if text
                    else ""
                )

        return ""

    @staticmethod
    def _extract_stream_finish_reason(
        event: Any,
    ) -> str | None:
        """Extract stop reason from stream events."""

        event_type = getattr(
            event,
            "type",
            None,
        )

        if event_type != "message_delta":
            return None

        delta = getattr(
            event,
            "delta",
            None,
        )

        if delta is None:
            return None

        reason = getattr(
            delta,
            "stop_reason",
            None,
        )

        if reason is None:
            return None

        return str(reason)

    # ========================================================
    # ERROR TRANSLATION
    # ========================================================

    @staticmethod
    def _translate_error(
        error: Exception,
    ) -> Exception:
        """
        Translate common Anthropic SDK errors into RENIX
        provider exceptions.
        """

        name = type(error).__name__

        message = str(error)

        lowered = message.lower()

        if (
            "authentication" in lowered
            or "api key" in lowered
            or "invalid x-api-key" in lowered
            or "permission denied" in lowered
            or name.lower()
            in {
                "authenticationerror",
                "permissionerror",
            }
        ):
            return ProviderAuthenticationError(
                message
            )

        if (
            "rate limit" in lowered
            or "too many requests" in lowered
            or "overloaded" in lowered
        ):
            return ProviderResponseError(
                f"Anthropic rate limit/overload: "
                f"{message}"
            )

        if (
            "timeout" in lowered
            or "timed out" in lowered
        ):
            return ProviderTimeoutError(
                message
            )

        if (
            "connection" in lowered
            or "network" in lowered
        ):
            return ProviderConnectionError(
                message
            )

        if (
            "not found" in lowered
            or "unsupported" in lowered
            or "not supported" in lowered
        ):
            return ProviderNotSupportedError(
                message
            )

        if (
            "invalid" in lowered
            or "bad request" in lowered
        ):
            return ProviderResponseError(
                message
            )

        return ProviderResponseError(
            message
        )


# ============================================================
# FACTORY
# ============================================================


def create_anthropic_provider(
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = 60.0,
    max_retries: int = 2,
    client: Any | None = None,
) -> AnthropicProvider:
    """
    Create an AnthropicProvider using explicit values or
    environment variables.
    """

    config = ProviderConfig(
        name="anthropic",
        api_key=(
            api_key
            or os.getenv(
                "ANTHROPIC_API_KEY"
            )
        ),
        base_url=(
            base_url
            or os.getenv(
                "ANTHROPIC_BASE_URL"
            )
        ),
        default_model=(
            model
            or os.getenv(
                "RENIX_ANTHROPIC_MODEL",
                "claude-sonnet-4-5",
            )
        ),
        timeout=timeout,
        max_retries=max_retries,
    )

    return AnthropicProvider(
        config,
        client=client,
    )


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "AnthropicProvider",
    "create_anthropic_provider",
]


