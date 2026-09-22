"""
RENIX OpenAI Provider
=====================

OpenAI implementation of the RENIX AIProvider interface.

This module keeps OpenAI-specific behavior isolated from the
rest of RENIX.

API credentials should be supplied through environment
variables or RENIX's secrets manager.

Expected environment variable by default:

    OPENAI_API_KEY

Optional:

    OPENAI_BASE_URL
    OPENAI_ORGANIZATION

The OpenAI SDK is imported lazily so that RENIX can still
start when the OpenAI package is not installed.
"""

from __future__ import annotations

import os
import time
import uuid

from typing import Any, Iterator, Mapping

from .provider import (
    AIProvider,
    AIRequest,
    AIResponse,
    AIStreamChunk,
    ProviderAuthenticationError,
    ProviderCapabilities,
    ProviderConfig,
    ProviderConnectionError,
    ProviderResponseError,
    ProviderStatus,
    ProviderTimeoutError,
    ResponseType,
)


# ============================================================
# OPENAI PROVIDER
# ============================================================


class OpenAIProvider(AIProvider):
    """
    RENIX provider implementation for OpenAI-compatible APIs.

    The implementation uses the modern OpenAI Python SDK when
    available.

    Example:

        config = ProviderConfig(
            name="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            default_model="gpt-4o-mini",
        )

        provider = OpenAIProvider(config)
        provider.initialize()

        response = provider.generate(
            AIRequest(
                prompt="Hello RENIX"
            )
        )

        print(response.text)
    """

    def __init__(
        self,
        config: ProviderConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:

        if config is None:
            config = ProviderConfig(
                name="openai",
                api_key=os.getenv(
                    "OPENAI_API_KEY"
                ),
                base_url=os.getenv(
                    "OPENAI_BASE_URL"
                ),
                organization=os.getenv(
                    "OPENAI_ORGANIZATION"
                ),
                default_model=os.getenv(
                    "RENIX_OPENAI_MODEL",
                    "gpt-4o-mini",
                ),
            )

        capabilities = ProviderCapabilities(
            chat=True,
            streaming=True,
            vision=True,
            audio_input=True,
            audio_output=True,
            embeddings=True,
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

        return "openai"

    # ========================================================
    # CLIENT
    # ========================================================

    def _create_client(self) -> Any:
        """
        Lazily create the OpenAI client.

        This avoids importing the SDK until the provider is
        actually used.
        """

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderConnectionError(
                "The OpenAI Python SDK is not installed. "
                "Install the 'openai' package."
            ) from exc

        api_key = self.config.api_key

        if not api_key:
            raise ProviderAuthenticationError(
                "OpenAI API key is not configured. "
                "Set OPENAI_API_KEY or provide it through "
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

        if self.config.organization:
            kwargs["organization"] = (
                self.config.organization
            )

        try:
            return OpenAI(**kwargs)
        except Exception as exc:
            raise ProviderConnectionError(
                f"Failed to initialize OpenAI client: "
                f"{exc}"
            ) from exc

    @property
    def client(self) -> Any:
        """Return lazily initialized OpenAI client."""

        if self._client is None:
            self._client = self._create_client()

        return self._client

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def initialize(self) -> None:
        """Initialize the OpenAI provider."""

        if not self.enabled:
            self.status = (
                ProviderStatus.DISABLED
            )
            return

        # Do not make a network request merely to initialize.
        # The client is created lazily.
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
        Perform a lightweight provider availability check.

        The call intentionally uses the models endpoint instead
        of generating text, avoiding unnecessary token usage.
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
            self.client.models.list()

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
    ) -> list[dict[str, str]]:
        """Build OpenAI-compatible message objects."""

        messages: list[
            dict[str, str]
        ] = []

        if request.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": request.system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": request.prompt,
            }
        )

        return messages

    def _build_tools(
        self,
        request: AIRequest,
    ) -> list[dict[str, Any]] | None:
        """Build tool definitions."""

        if not request.tools:
            return None

        return [
            dict(tool)
            for tool in request.tools
        ]

    def _build_request_kwargs(
        self,
        request: AIRequest,
    ) -> dict[str, Any]:
        """Convert AIRequest to OpenAI parameters."""

        model = self.resolve_model(
            request.model
        )

        if not model:
            raise ProviderResponseError(
                "No OpenAI model was specified."
            )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._build_messages(
                request
            ),
        }

        if request.temperature is not None:
            kwargs["temperature"] = (
                request.temperature
            )

        if request.max_tokens is not None:
            # The current SDK/API may prefer max_completion_tokens
            # for newer models. We use it here.
            kwargs["max_completion_tokens"] = (
                request.max_tokens
            )

        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        if request.stop_sequences:
            kwargs["stop"] = list(
                request.stop_sequences
            )

        tools = self._build_tools(
            request
        )

        if tools:
            kwargs["tools"] = tools

        if request.response_type == ResponseType.JSON:
            kwargs["response_format"] = {
                "type": "json_object"
            }

        return kwargs

    # ========================================================
    # RESPONSE PARSING
    # ========================================================

    @staticmethod
    def _extract_text(
        response: Any,
    ) -> str:
        """Extract assistant text from an SDK response."""

        try:
            choices = response.choices

            if not choices:
                return ""

            message = choices[0].message

            content = getattr(
                message,
                "content",
                None,
            )

            if content is None:
                return ""

            return str(content)

        except (
            AttributeError,
            IndexError,
            TypeError,
        ):
            return ""

    @staticmethod
    def _extract_tool_calls(
        response: Any,
    ) -> list[dict[str, Any]]:
        """Extract tool calls from an SDK response."""

        result: list[
            dict[str, Any]
        ] = []

        try:
            choices = response.choices

            if not choices:
                return result

            message = choices[0].message

            tool_calls = getattr(
                message,
                "tool_calls",
                None,
            )

            if not tool_calls:
                return result

            for call in tool_calls:
                function = getattr(
                    call,
                    "function",
                    None,
                )

                result.append(
                    {
                        "id": getattr(
                            call,
                            "id",
                            None,
                        ),
                        "type": getattr(
                            call,
                            "type",
                            "function",
                        ),
                        "name": getattr(
                            function,
                            "name",
                            None,
                        ),
                        "arguments": getattr(
                            function,
                            "arguments",
                            None,
                        ),
                    }
                )

        except (
            AttributeError,
            TypeError,
        ):
            pass

        return result

    @staticmethod
    def _extract_usage(
        response: Any,
    ) -> tuple[int, int, int]:
        """Extract token usage."""

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
                "prompt_tokens",
                0,
            )
            or 0
        )

        output_tokens = int(
            getattr(
                usage,
                "completion_tokens",
                0,
            )
            or 0
        )

        total_tokens = int(
            getattr(
                usage,
                "total_tokens",
                input_tokens
                + output_tokens,
            )
            or 0
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
        """Extract provider request ID."""

        value = getattr(
            response,
            "id",
            None,
        )

        if value is None:
            return None

        return str(value)

    @staticmethod
    def _extract_finish_reason(
        response: Any,
    ) -> str | None:
        """Extract finish reason."""

        try:
            reason = (
                response
                .choices[0]
                .finish_reason
            )

            if reason is None:
                return None

            return str(reason)

        except (
            AttributeError,
            IndexError,
        ):
            return None

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        request: AIRequest,
    ) -> AIResponse:
        """
        Generate a complete response through OpenAI.
        """

        self.validate_request(
            request
        )

        self.record_request()

        started = time.perf_counter()

        try:
            response = (
                self.client.chat.completions.create(
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
                    "provider": "openai",
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
                f"OpenAI request timed out: "
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
        """
        Stream an OpenAI response.
        """

        self.validate_request(
            request
        )

        if not self.capabilities.streaming:
            raise ProviderResponseError(
                "OpenAI streaming is unavailable."
            )

        self.record_request()

        self.metrics.streamed_requests += 1

        started = time.perf_counter()

        try:
            kwargs = self._build_request_kwargs(
                request
            )

            kwargs["stream"] = True

            stream = (
                self.client.chat.completions.create(
                    **kwargs
                )
            )

            request_id: str | None = None

            for index, chunk in enumerate(
                stream
            ):
                request_id = (
                    request_id
                    or self._extract_request_id(
                        chunk
                    )
                )

                text = (
                    self._extract_stream_text(
                        chunk
                    )
                )

                finish_reason = (
                    self._extract_stream_finish_reason(
                        chunk
                    )
                )

                is_final = (
                    finish_reason is not None
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
                        "provider": "openai"
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
                f"OpenAI streaming request timed out: "
                f"{exc}"
            ) from exc

        except Exception as exc:
            self.record_error(exc)

            raise self._translate_error(
                exc
            ) from exc

    # ========================================================
    # STREAM PARSING
    # ========================================================

    @staticmethod
    def _extract_stream_text(
        chunk: Any,
    ) -> str:
        """Extract text from a streaming chunk."""

        try:
            choices = chunk.choices

            if not choices:
                return ""

            delta = choices[0].delta

            content = getattr(
                delta,
                "content",
                None,
            )

            if content is None:
                return ""

            return str(content)

        except (
            AttributeError,
            IndexError,
            TypeError,
        ):
            return ""

    @staticmethod
    def _extract_stream_finish_reason(
        chunk: Any,
    ) -> str | None:
        """Extract streaming finish reason."""

        try:
            reason = (
                chunk
                .choices[0]
                .finish_reason
            )

            if reason is None:
                return None

            return str(reason)

        except (
            AttributeError,
            IndexError,
        ):
            return None

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    def embed(
        self,
        text: str,
        *,
        model: str | None = None,
    ) -> list[float]:
        """Generate an OpenAI embedding."""

        if not text or not str(text).strip():
            raise ValueError(
                "Embedding text cannot be empty."
            )

        self.record_request()

        try:
            embedding_model = (
                model
                or self.config.extra.get(
                    "embedding_model",
                    "text-embedding-3-small",
                )
            )

            response = (
                self.client.embeddings.create(
                    model=embedding_model,
                    input=text,
                )
            )

            self.status = (
                ProviderStatus.AVAILABLE
            )

            if not response.data:
                raise ProviderResponseError(
                    "OpenAI returned no embedding data."
                )

            embedding = response.data[0].embedding

            return [
                float(value)
                for value in embedding
            ]

        except Exception as exc:
            self.record_error(exc)

            raise self._translate_error(
                exc
            ) from exc

    # ========================================================
    # ERROR TRANSLATION
    # ========================================================

    @staticmethod
    def _translate_error(
        error: Exception,
    ) -> Exception:
        """
        Convert common OpenAI SDK exceptions into RENIX
        provider exceptions.

        The implementation intentionally avoids depending on
        specific SDK exception classes so this module remains
        compatible across SDK versions.
        """

        name = type(error).__name__

        message = str(error)

        lowered = message.lower()

        if (
            "authentication" in lowered
            or "api key" in lowered
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
            or name.lower()
            == "ratelimiterror"
        ):
            return ProviderResponseError(
                f"OpenAI rate limit reached: "
                f"{message}"
            )

        if (
            "timeout" in lowered
            or "timed out" in lowered
            or name.lower()
            in {
                "timeouterror",
                "apitimeouterror",
            }
        ):
            return ProviderTimeoutError(
                message
            )

        if (
            "connection" in lowered
            or "connect" in lowered
            or name.lower()
            in {
                "connectionerror",
                "apiconnectionerror",
            }
        ):
            return ProviderConnectionError(
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


def create_openai_provider(
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
    timeout: float = 60.0,
    max_retries: int = 2,
    client: Any | None = None,
) -> OpenAIProvider:
    """
    Create an OpenAIProvider from explicit parameters or
    environment variables.

    The API key is never written to disk by this function.
    """

    config = ProviderConfig(
        name="openai",
        api_key=(
            api_key
            or os.getenv(
                "OPENAI_API_KEY"
            )
        ),
        base_url=(
            base_url
            or os.getenv(
                "OPENAI_BASE_URL"
            )
        ),
        organization=(
            organization
            or os.getenv(
                "OPENAI_ORGANIZATION"
            )
        ),
        default_model=(
            model
            or os.getenv(
                "RENIX_OPENAI_MODEL",
                "gpt-4o-mini",
            )
        ),
        timeout=timeout,
        max_retries=max_retries,
    )

    return OpenAIProvider(
        config,
        client=client,
    )


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "OpenAIProvider",
    "create_openai_provider",
]


