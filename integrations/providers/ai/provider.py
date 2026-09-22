"""
RENIX AI Provider Base
======================

Defines the common interface that every external AI provider
used by RENIX should implement.

Provider-specific implementations can inherit from
AIProvider and implement the required methods.

Examples of providers that can follow this interface:

    - OpenAI
    - Google Gemini
    - Anthropic
    - Ollama
    - Local models
    - Future custom providers

This module intentionally contains no provider-specific API
calls. It provides the abstraction layer used by RENIX.
"""

from __future__ import annotations

import time

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Iterator, Mapping


# ============================================================
# ENUMS
# ============================================================


class ProviderStatus(str, Enum):
    """Current state of an AI provider."""

    UNKNOWN = "unknown"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    DISABLED = "disabled"


class ResponseType(str, Enum):
    """Supported response types."""

    TEXT = "text"
    JSON = "json"
    STREAM = "stream"
    TOOL_CALL = "tool_call"


# ============================================================
# EXCEPTIONS
# ============================================================


class AIProviderError(Exception):
    """Base exception for AI provider errors."""


class ProviderConfigurationError(
    AIProviderError
):
    """Invalid provider configuration."""


class ProviderAuthenticationError(
    AIProviderError
):
    """Provider authentication failed."""


class ProviderConnectionError(
    AIProviderError
):
    """Provider connection failed."""


class ProviderRateLimitError(
    AIProviderError
):
    """Provider rate limit was reached."""


class ProviderTimeoutError(
    AIProviderError
):
    """Provider request timed out."""


class ProviderResponseError(
    AIProviderError
):
    """Provider returned an invalid response."""


class ProviderNotSupportedError(
    AIProviderError
):
    """Requested provider capability is unsupported."""


# ============================================================
# REQUEST
# ============================================================


@dataclass
class AIRequest:
    """
    Standard RENIX AI request.

    This object allows RENIX's internal systems to send
    provider-independent requests.
    """

    prompt: str

    system_prompt: str | None = None

    model: str | None = None

    temperature: float | None = None

    max_tokens: int | None = None

    top_p: float | None = None

    stop_sequences: list[str] = field(
        default_factory=list
    )

    conversation_id: str | None = None

    user_id: str | None = None

    tools: list[dict[str, Any]] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    stream: bool = False

    response_type: ResponseType = (
        ResponseType.TEXT
    )

    timeout: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.prompt,
            str,
        ):
            raise TypeError(
                "prompt must be a string."
            )

        self.prompt = self.prompt.strip()

        if not self.prompt:
            raise ValueError(
                "prompt cannot be empty."
            )

        if self.temperature is not None:
            if not 0 <= self.temperature <= 2:
                raise ValueError(
                    "temperature must be between "
                    "0 and 2."
                )

        if self.top_p is not None:
            if not 0 <= self.top_p <= 1:
                raise ValueError(
                    "top_p must be between "
                    "0 and 1."
                )

        if (
            self.max_tokens is not None
            and self.max_tokens <= 0
        ):
            raise ValueError(
                "max_tokens must be positive."
            )

        if (
            self.timeout is not None
            and self.timeout <= 0
        ):
            raise ValueError(
                "timeout must be positive."
            )

    def to_dict(
        self,
        *,
        include_prompt: bool = True,
    ) -> dict[str, Any]:
        """Convert request to a serializable dictionary."""

        data = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop_sequences": list(
                self.stop_sequences
            ),
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "tools": list(self.tools),
            "metadata": dict(self.metadata),
            "stream": self.stream,
            "response_type": self.response_type.value,
            "timeout": self.timeout,
        }

        if include_prompt:
            data["prompt"] = self.prompt
            data["system_prompt"] = (
                self.system_prompt
            )

        return data


# ============================================================
# RESPONSE
# ============================================================


@dataclass
class AIResponse:
    """
    Standardized RENIX AI response.
    """

    text: str = ""

    model: str | None = None

    provider: str | None = None

    request_id: str | None = None

    finish_reason: str | None = None

    input_tokens: int = 0

    output_tokens: int = 0

    total_tokens: int = 0

    latency_ms: float = 0.0

    tool_calls: list[dict[str, Any]] = field(
        default_factory=list
    )

    raw_response: Any = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: float = field(
        default_factory=time.time
    )

    success: bool = True

    error: str | None = None

    def __post_init__(self) -> None:
        if self.total_tokens == 0:
            self.total_tokens = (
                self.input_tokens
                + self.output_tokens
            )

    def to_dict(
        self,
        *,
        include_raw: bool = False,
    ) -> dict[str, Any]:
        """Convert response to a dictionary."""

        data = {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "request_id": self.request_id,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "tool_calls": list(
                self.tool_calls
            ),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "success": self.success,
            "error": self.error,
        }

        if include_raw:
            data["raw_response"] = (
                self.raw_response
            )

        return data


# ============================================================
# STREAM CHUNK
# ============================================================


@dataclass
class AIStreamChunk:
    """
    Standardized streaming response chunk.
    """

    text: str = ""

    provider: str | None = None

    model: str | None = None

    request_id: str | None = None

    index: int = 0

    is_final: bool = False

    finish_reason: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize a stream chunk."""

        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "request_id": self.request_id,
            "index": self.index,
            "is_final": self.is_final,
            "finish_reason": self.finish_reason,
            "metadata": dict(self.metadata),
        }


# ============================================================
# PROVIDER CAPABILITIES
# ============================================================


@dataclass
class ProviderCapabilities:
    """
    Describes what an AI provider supports.
    """

    chat: bool = True

    streaming: bool = False

    vision: bool = False

    audio_input: bool = False

    audio_output: bool = False

    embeddings: bool = False

    tool_calling: bool = False

    structured_output: bool = False

    image_generation: bool = False

    fine_tuning: bool = False

    reasoning: bool = False

    max_context_tokens: int | None = None

    supported_models: list[str] = field(
        default_factory=list
    )

    extra: dict[str, Any] = field(
        default_factory=dict
    )

    def supports_model(
        self,
        model: str,
    ) -> bool:
        """
        Determine whether a model is explicitly supported.

        An empty model list means the provider does not
        maintain a static model list.
        """

        if not self.supported_models:
            return True

        return model in self.supported_models

    def to_dict(self) -> dict[str, Any]:
        """Serialize capabilities."""

        return {
            "chat": self.chat,
            "streaming": self.streaming,
            "vision": self.vision,
            "audio_input": self.audio_input,
            "audio_output": self.audio_output,
            "embeddings": self.embeddings,
            "tool_calling": self.tool_calling,
            "structured_output": self.structured_output,
            "image_generation": self.image_generation,
            "fine_tuning": self.fine_tuning,
            "reasoning": self.reasoning,
            "max_context_tokens": (
                self.max_context_tokens
            ),
            "supported_models": list(
                self.supported_models
            ),
            "extra": dict(self.extra),
        }


# ============================================================
# PROVIDER CONFIGURATION
# ============================================================


@dataclass
class ProviderConfig:
    """
    Generic provider configuration.

    Secrets should normally be injected through RENIX's
    secrets manager rather than committed to configuration
    files.
    """

    name: str

    api_key: str | None = None

    base_url: str | None = None

    default_model: str | None = None

    enabled: bool = True

    timeout: float = 60.0

    max_retries: int = 2

    organization: str | None = None

    project: str | None = None

    extra: dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        """Validate provider configuration."""

        if not self.name.strip():
            raise ProviderConfigurationError(
                "Provider name cannot be empty."
            )

        if self.timeout <= 0:
            raise ProviderConfigurationError(
                "Provider timeout must be positive."
            )

        if self.max_retries < 0:
            raise ProviderConfigurationError(
                "max_retries cannot be negative."
            )

    def public_dict(self) -> dict[str, Any]:
        """
        Return safe configuration information.

        API keys are intentionally excluded.
        """

        return {
            "name": self.name,
            "base_url": self.base_url,
            "default_model": self.default_model,
            "enabled": self.enabled,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "organization": self.organization,
            "project": self.project,
            "extra": dict(self.extra),
        }


# ============================================================
# PROVIDER METRICS
# ============================================================


@dataclass
class ProviderMetrics:
    """
    Runtime statistics for a provider.
    """

    requests: int = 0

    successful_requests: int = 0

    failed_requests: int = 0

    streamed_requests: int = 0

    total_input_tokens: int = 0

    total_output_tokens: int = 0

    total_latency_ms: float = 0.0

    last_request_at: float | None = None

    last_success_at: float | None = None

    last_error_at: float | None = None

    last_error: str | None = None

    def record_request(
        self,
    ) -> None:
        """Record a request."""

        self.requests += 1
        self.last_request_at = time.time()

    def record_success(
        self,
        response: AIResponse,
    ) -> None:
        """Record successful response."""

        self.successful_requests += 1

        self.total_input_tokens += (
            response.input_tokens
        )

        self.total_output_tokens += (
            response.output_tokens
        )

        self.total_latency_ms += (
            response.latency_ms
        )

        self.last_success_at = time.time()

        self.last_error = None

    def record_failure(
        self,
        error: Exception | str,
    ) -> None:
        """Record failed request."""

        self.failed_requests += 1

        self.last_error_at = time.time()

        self.last_error = str(error)

    @property
    def average_latency_ms(self) -> float:
        """Calculate average successful-request latency."""

        if self.successful_requests == 0:
            return 0.0

        return (
            self.total_latency_ms
            / self.successful_requests
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics."""

        return {
            "requests": self.requests,
            "successful_requests": (
                self.successful_requests
            ),
            "failed_requests": (
                self.failed_requests
            ),
            "streamed_requests": (
                self.streamed_requests
            ),
            "total_input_tokens": (
                self.total_input_tokens
            ),
            "total_output_tokens": (
                self.total_output_tokens
            ),
            "total_latency_ms": (
                self.total_latency_ms
            ),
            "average_latency_ms": (
                self.average_latency_ms
            ),
            "last_request_at": (
                self.last_request_at
            ),
            "last_success_at": (
                self.last_success_at
            ),
            "last_error_at": (
                self.last_error_at
            ),
            "last_error": self.last_error,
        }


# ============================================================
# ABSTRACT AI PROVIDER
# ============================================================


class AIProvider(ABC):
    """
    Base class for every RENIX AI provider.

    Provider implementations should subclass this class.

    Required methods:

        - generate()
        - stream()

    Optional methods can be implemented when supported:

        - embed()
        - generate_image()
        - transcribe()
        - synthesize()
    """

    def __init__(
        self,
        config: ProviderConfig,
        capabilities: ProviderCapabilities | None = None,
    ) -> None:

        if not isinstance(
            config,
            ProviderConfig,
        ):
            raise TypeError(
                "config must be ProviderConfig."
            )

        config.validate()

        self.config = config

        self.capabilities = (
            capabilities
            or ProviderCapabilities()
        )

        self.status = (
            ProviderStatus.UNKNOWN
        )

        self.metrics = ProviderMetrics()

        self._initialized = False

    # ========================================================
    # PROPERTIES
    # ========================================================

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Return provider identifier.

        Example:
            "openai"
            "gemini"
            "anthropic"
            "ollama"
        """

        raise NotImplementedError

    @property
    def default_model(self) -> str | None:
        """Return configured default model."""

        return self.config.default_model

    @property
    def enabled(self) -> bool:
        """Return whether provider is enabled."""

        return self.config.enabled

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def initialize(self) -> None:
        """
        Initialize provider resources.

        Subclasses may override this method.
        """

        if not self.enabled:
            self.status = (
                ProviderStatus.DISABLED
            )
            return

        self._initialized = True

        self.status = (
            ProviderStatus.UNKNOWN
        )

    def shutdown(self) -> None:
        """
        Shut down provider resources.

        Subclasses may override this method.
        """

        self._initialized = False

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(self) -> bool:
        """
        Check provider availability.

        Subclasses should override this method with a real
        provider health check.
        """

        if not self.enabled:
            self.status = (
                ProviderStatus.DISABLED
            )
            return False

        self.status = (
            ProviderStatus.AVAILABLE
        )

        return True

    def health_status(self) -> dict[str, Any]:
        """Return safe health information."""

        return {
            "provider": self.provider_name,
            "status": self.status.value,
            "enabled": self.enabled,
            "initialized": self._initialized,
            "model": self.default_model,
            "capabilities": (
                self.capabilities.to_dict()
            ),
            "metrics": (
                self.metrics.to_dict()
            ),
        }

    # ========================================================
    # REQUEST VALIDATION
    # ========================================================

    def validate_request(
        self,
        request: AIRequest,
    ) -> None:
        """Validate a standardized AI request."""

        if not isinstance(
            request,
            AIRequest,
        ):
            raise TypeError(
                "request must be an AIRequest."
            )

        if not self.enabled:
            raise AIProviderError(
                f"Provider '{self.provider_name}' "
                "is disabled."
            )

        if not self.capabilities.chat:
            raise ProviderNotSupportedError(
                f"Provider '{self.provider_name}' "
                "does not support chat generation."
            )

        model = (
            request.model
            or self.default_model
        )

        if model and not self.capabilities.supports_model(
            model
        ):
            raise ProviderNotSupportedError(
                f"Model '{model}' is not supported "
                f"by provider '{self.provider_name}'."
            )

        if (
            request.stream
            and not self.capabilities.streaming
        ):
            raise ProviderNotSupportedError(
                f"Provider '{self.provider_name}' "
                "does not support streaming."
            )

        if (
            request.tools
            and not self.capabilities.tool_calling
        ):
            raise ProviderNotSupportedError(
                f"Provider '{self.provider_name}' "
                "does not support tool calling."
            )

    # ========================================================
    # GENERATION
    # ========================================================

    @abstractmethod
    def generate(
        self,
        request: AIRequest,
    ) -> AIResponse:
        """
        Generate a complete response.

        Every concrete AI provider must implement this.
        """

        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        request: AIRequest,
    ) -> Iterator[AIStreamChunk]:
        """
        Stream a response.

        Every concrete AI provider must implement this.
        """

        raise NotImplementedError

    async def stream_async(
        self,
        request: AIRequest,
    ) -> AsyncIterator[AIStreamChunk]:
        """
        Async streaming interface.

        Providers with native async support should override
        this method.
        """

        for chunk in self.stream(request):
            yield chunk

    # ========================================================
    # OPTIONAL CAPABILITIES
    # ========================================================

    def embed(
        self,
        text: str,
        *,
        model: str | None = None,
    ) -> list[float]:
        """Generate an embedding."""

        if not self.capabilities.embeddings:
            raise ProviderNotSupportedError(
                f"Provider '{self.provider_name}' "
                "does not support embeddings."
            )

        raise NotImplementedError(
            "Embedding support has not been implemented "
            "by this provider."
        )

    def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate an image."""

        if not self.capabilities.image_generation:
            raise ProviderNotSupportedError(
                f"Provider '{self.provider_name}' "
                "does not support image generation."
            )

        raise NotImplementedError(
            "Image generation has not been implemented "
            "by this provider."
        )

    def transcribe(
        self,
        audio: bytes,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Transcribe audio."""

        if not self.capabilities.audio_input:
            raise ProviderNotSupportedError(
                f"Provider '{self.provider_name}' "
                "does not support audio input."
            )

        raise NotImplementedError(
            "Audio transcription has not been implemented "
            "by this provider."
        )

    def synthesize(
        self,
        text: str,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Synthesize speech."""

        if not self.capabilities.audio_output:
            raise ProviderNotSupportedError(
                f"Provider '{self.provider_name}' "
                "does not support audio output."
            )

        raise NotImplementedError(
            "Speech synthesis has not been implemented "
            "by this provider."
        )

    # ========================================================
    # METRICS HELPERS
    # ========================================================

    def record_request(
        self,
    ) -> None:
        """Record the beginning of a request."""

        self.metrics.record_request()

    def record_response(
        self,
        response: AIResponse,
    ) -> None:
        """Record a successful response."""

        self.metrics.record_success(
            response
        )

    def record_error(
        self,
        error: Exception | str,
    ) -> None:
        """Record a failed request."""

        self.metrics.record_failure(
            error
        )

        self.status = (
            ProviderStatus.ERROR
        )

    # ========================================================
    # MODEL HELPERS
    # ========================================================

    def resolve_model(
        self,
        requested_model: str | None,
    ) -> str | None:
        """
        Resolve requested model or provider default.
        """

        return (
            requested_model
            or self.default_model
        )

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def info(self) -> dict[str, Any]:
        """
        Return provider information without secrets.
        """

        return {
            "name": self.provider_name,
            "config": (
                self.config.public_dict()
            ),
            "capabilities": (
                self.capabilities.to_dict()
            ),
            "status": self.status.value,
            "initialized": self._initialized,
            "metrics": (
                self.metrics.to_dict()
            ),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.provider_name!r}, "
            f"model={self.default_model!r}, "
            f"status={self.status.value!r})"
        )


# ============================================================
# NULL PROVIDER
# ============================================================


class NullAIProvider(AIProvider):
    """
    Safe placeholder provider.

    Useful when RENIX starts without an external AI provider.
    It never contacts an external service.
    """

    @property
    def provider_name(self) -> str:
        return "null"

    def __init__(
        self,
    ) -> None:

        super().__init__(
            ProviderConfig(
                name="null",
                enabled=True,
            ),
            ProviderCapabilities(
                chat=True,
                streaming=True,
            ),
        )

        self.status = (
            ProviderStatus.AVAILABLE
        )

    def generate(
        self,
        request: AIRequest,
    ) -> AIResponse:

        self.validate_request(
            request
        )

        self.record_request()

        response = AIResponse(
            text=(
                "RENIX AI provider is not configured yet."
            ),
            model=None,
            provider=self.provider_name,
            finish_reason="provider_not_configured",
            metadata={
                "null_provider": True
            },
        )

        self.record_response(
            response
        )

        return response

    def stream(
        self,
        request: AIRequest,
    ) -> Iterator[AIStreamChunk]:

        self.validate_request(
            request
        )

        self.record_request()

        text = (
            "RENIX AI provider is not configured yet."
        )

        yield AIStreamChunk(
            text=text,
            provider=self.provider_name,
            model=None,
            index=0,
            is_final=False,
        )

        yield AIStreamChunk(
            text="",
            provider=self.provider_name,
            model=None,
            index=1,
            is_final=True,
            finish_reason=(
                "provider_not_configured"
            ),
        )


# ============================================================
# FACTORY
# ============================================================


def create_null_provider() -> NullAIProvider:
    """Create the default null provider."""

    return NullAIProvider()


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "AIProvider",
    "AIRequest",
    "AIResponse",
    "AIStreamChunk",
    "ProviderCapabilities",
    "ProviderConfig",
    "ProviderMetrics",
    "ProviderStatus",
    "ResponseType",
    "NullAIProvider",
    "create_null_provider",
    "AIProviderError",
    "ProviderConfigurationError",
    "ProviderAuthenticationError",
    "ProviderConnectionError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderResponseError",
    "ProviderNotSupportedError",
]


