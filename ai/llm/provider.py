"""
RENIX AI
LLM Provider Layer

Defines the common provider interface used by RENIX to communicate
with different Large Language Model providers.

This module does not hard-code a specific AI provider. Providers
can implement the LLMProvider interface and be selected by the
RENIX LLM router.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class LLMMessage:
    """
    Represents one message sent to an LLM.
    """

    role: str
    content: str

    name: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message into a provider-friendly dictionary.
        """

        data: Dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }

        if self.name:
            data["name"] = self.name

        if self.metadata:
            data["metadata"] = dict(self.metadata)

        return data


@dataclass
class LLMRequest:
    """
    Standard RENIX request sent to an LLM provider.
    """

    messages: List[LLMMessage]

    model: Optional[str] = None

    temperature: float = 0.7

    max_tokens: Optional[int] = None

    top_p: Optional[float] = None

    stream: bool = False

    stop: Optional[List[str]] = None

    system_prompt: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    tools: Optional[List[Dict[str, Any]]] = None

    response_format: Optional[Dict[str, Any]] = None

    timeout: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the request into a serializable dictionary.
        """

        data: Dict[str, Any] = {
            "messages": [
                message.to_dict()
                for message in self.messages
            ],
            "temperature": self.temperature,
            "stream": self.stream,
        }

        if self.model is not None:
            data["model"] = self.model

        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens

        if self.top_p is not None:
            data["top_p"] = self.top_p

        if self.stop is not None:
            data["stop"] = list(self.stop)

        if self.system_prompt is not None:
            data["system_prompt"] = self.system_prompt

        if self.metadata:
            data["metadata"] = dict(self.metadata)

        if self.tools is not None:
            data["tools"] = list(self.tools)

        if self.response_format is not None:
            data["response_format"] = dict(
                self.response_format
            )

        if self.timeout is not None:
            data["timeout"] = self.timeout

        return data


@dataclass
class LLMResponse:
    """
    Standard RENIX response returned by an LLM provider.
    """

    content: str

    model: Optional[str] = None

    provider: Optional[str] = None

    finish_reason: Optional[str] = None

    prompt_tokens: int = 0

    completion_tokens: int = 0

    total_tokens: int = 0

    response_id: Optional[str] = None

    tool_calls: List[Dict[str, Any]] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    raw_response: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the response into a serializable dictionary.
        """

        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "response_id": self.response_id,
            "tool_calls": list(self.tool_calls),
            "metadata": dict(self.metadata),
        }


@dataclass
class LLMChunk:
    """
    Represents one streaming chunk from an LLM.
    """

    content: str = ""

    model: Optional[str] = None

    provider: Optional[str] = None

    finish_reason: Optional[str] = None

    response_id: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    is_final: bool = False


@dataclass
class ProviderCapabilities:
    """
    Describes capabilities supported by an LLM provider.
    """

    streaming: bool = False

    tool_calling: bool = False

    vision: bool = False

    image_generation: bool = False

    embeddings: bool = False

    reasoning: bool = False

    structured_output: bool = False

    audio_input: bool = False

    audio_output: bool = False

    multilingual: bool = True

    max_context_tokens: Optional[int] = None

    supported_models: List[str] = field(
        default_factory=list
    )


# ============================================================================
# PROVIDER INTERFACE
# ============================================================================


class LLMProvider(ABC):
    """
    Abstract base class for every RENIX LLM provider.

    A provider is responsible for translating the standard RENIX
    request format into the API format required by its backend.
    """

    name: str = "unknown"

    version: str = "1.0"

    capabilities: ProviderCapabilities = (
        ProviderCapabilities()
    )

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.api_key = api_key

        self.default_model = default_model

        self.config: Dict[str, Any] = dict(
            config or {}
        )

        self._initialized = False

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def initialize(self) -> None:
        """
        Initialize the provider.

        Subclasses can override this when they need to create
        clients, sessions, or other resources.
        """

        self._initialized = True

    def shutdown(self) -> None:
        """
        Shut down the provider and release resources.
        """

        self._initialized = False

    @property
    def initialized(self) -> bool:
        """
        Return whether the provider is initialized.
        """

        return self._initialized

    # ========================================================================
    # IDENTIFICATION
    # ========================================================================

    def get_name(self) -> str:
        """
        Return the provider name.
        """

        return self.name

    def get_version(self) -> str:
        """
        Return the provider implementation version.
        """

        return self.version

    def get_capabilities(self) -> ProviderCapabilities:
        """
        Return provider capabilities.
        """

        return self.capabilities

    def supports(
        self,
        capability: str,
    ) -> bool:
        """
        Check whether a provider supports a specific capability.
        """

        return bool(
            getattr(
                self.capabilities,
                capability,
                False,
            )
        )

    # ========================================================================
    # MODEL MANAGEMENT
    # ========================================================================

    def get_default_model(self) -> Optional[str]:
        """
        Return the default model.
        """

        return self.default_model

    def set_default_model(
        self,
        model: str,
    ) -> None:
        """
        Change the default model.
        """

        if not model:
            raise ValueError(
                "Model name cannot be empty."
            )

        self.default_model = model

    def get_models(self) -> List[str]:
        """
        Return models known by the provider.
        """

        return list(
            self.capabilities.supported_models
        )

    def supports_model(
        self,
        model: str,
    ) -> bool:
        """
        Determine whether a model is known to the provider.

        Providers may override this method if model availability
        must be checked dynamically.
        """

        if not model:
            return False

        supported = self.get_models()

        if not supported:
            return True

        return model in supported

    # ========================================================================
    # REQUEST PREPARATION
    # ========================================================================

    def prepare_request(
        self,
        request: LLMRequest,
    ) -> LLMRequest:
        """
        Prepare a request before sending it to the provider.
        """

        if not request.messages:
            raise ValueError(
                "LLM request must contain at least one message."
            )

        if request.temperature < 0:
            raise ValueError(
                "Temperature cannot be negative."
            )

        if request.top_p is not None:
            if not 0 <= request.top_p <= 1:
                raise ValueError(
                    "top_p must be between 0 and 1."
                )

        if request.max_tokens is not None:
            if request.max_tokens <= 0:
                raise ValueError(
                    "max_tokens must be greater than zero."
                )

        if request.model is None:
            request.model = self.default_model

        return request

    # ========================================================================
    # GENERATION
    # ========================================================================

    @abstractmethod
    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate a complete response.

        Every concrete provider must implement this method.
        """

        raise NotImplementedError

    @abstractmethod
    def generate_stream(
        self,
        request: LLMRequest,
    ) -> Iterator[LLMChunk]:
        """
        Generate a streaming response.

        Every concrete provider that supports streaming should
        implement this method.
        """

        raise NotImplementedError

    async def generate_async(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Asynchronous generation interface.

        The default implementation executes the synchronous
        implementation in the current context. Providers with
        native async clients should override this method.
        """

        return self.generate(
            request
        )

    async def generate_stream_async(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[LLMChunk]:
        """
        Asynchronous streaming interface.

        Providers with native asynchronous streaming should
        override this method.
        """

        for chunk in self.generate_stream(
            request
        ):
            yield chunk

    # ========================================================================
    # SIMPLE TEXT INTERFACE
    # ========================================================================

    def chat(
        self,
        message: str,
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Convenience method for sending a simple user message.
        """

        request = LLMRequest(
            messages=[
                LLMMessage(
                    role="user",
                    content=message,
                )
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            metadata=dict(
                metadata or {}
            ),
        )

        return self.generate(
            self.prepare_request(request)
        )

    async def chat_async(
        self,
        message: str,
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Asynchronous convenience method for simple chat.
        """

        request = LLMRequest(
            messages=[
                LLMMessage(
                    role="user",
                    content=message,
                )
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            metadata=dict(
                metadata or {}
            ),
        )

        return await self.generate_async(
            self.prepare_request(request)
        )

    # ========================================================================
    # HEALTH
    # ========================================================================

    def health_check(self) -> bool:
        """
        Perform a lightweight provider health check.

        Providers can override this to perform an actual API
        health check.
        """

        return self.initialized

    def status(self) -> Dict[str, Any]:
        """
        Return provider status information.
        """

        return {
            "name": self.get_name(),
            "version": self.get_version(),
            "initialized": self.initialized,
            "default_model": self.default_model,
            "capabilities": {
                key: value
                for key, value in vars(
                    self.capabilities
                ).items()
            },
            "healthy": self.health_check(),
        }

    # ========================================================================
    # REPRESENTATION
    # ========================================================================

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"model={self.default_model!r}, "
            f"initialized={self.initialized!r}"
            f")"
        )


# ============================================================================
# BASIC PROVIDER ADAPTER
# ============================================================================


class FunctionLLMProvider(LLMProvider):
    """
    Lightweight provider adapter useful for local development,
    testing, and connecting a custom generation function.

    The supplied function receives an LLMRequest and must return
    either an LLMResponse or a string.
    """

    name = "function"

    capabilities = ProviderCapabilities(
        streaming=False,
        tool_calling=False,
        vision=False,
        reasoning=False,
        structured_output=False,
    )

    def __init__(
        self,
        function: Any,
        *,
        default_model: Optional[str] = None,
        provider_name: str = "function",
    ) -> None:

        super().__init__(
            default_model=default_model
        )

        if not callable(function):
            raise TypeError(
                "function must be callable."
            )

        self.function = function

        self.name = provider_name

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        request = self.prepare_request(
            request
        )

        result = self.function(
            request
        )

        if isinstance(
            result,
            LLMResponse,
        ):
            return result

        return LLMResponse(
            content=str(result),
            model=request.model,
            provider=self.name,
        )

    def generate_stream(
        self,
        request: LLMRequest,
    ) -> Iterator[LLMChunk]:

        response = self.generate(
            request
        )

        yield LLMChunk(
            content=response.content,
            model=response.model,
            provider=response.provider,
            finish_reason=response.finish_reason,
            response_id=response.response_id,
            is_final=True,
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def create_message(
    role: str,
    content: str,
    *,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> LLMMessage:
    """
    Create a standard RENIX LLM message.
    """

    return LLMMessage(
        role=role,
        content=content,
        name=name,
        metadata=dict(
            metadata or {}
        ),
    )


def create_request(
    messages: List[LLMMessage],
    *,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    system_prompt: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> LLMRequest:
    """
    Create a standard RENIX LLM request.
    """

    return LLMRequest(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        system_prompt=system_prompt,
        metadata=dict(
            metadata or {}
        ),
        tools=tools,
    )


def normalize_response(
    response: Any,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMResponse:
    """
    Convert common response formats into an LLMResponse.
    """

    if isinstance(
        response,
        LLMResponse,
    ):
        return response

    if isinstance(
        response,
        str,
    ):
        return LLMResponse(
            content=response,
            provider=provider,
            model=model,
        )

    if isinstance(
        response,
        dict,
    ):
        return LLMResponse(
            content=str(
                response.get(
                    "content",
                    response.get(
                        "text",
                        "",
                    ),
                )
            ),
            provider=response.get(
                "provider",
                provider,
            ),
            model=response.get(
                "model",
                model,
            ),
            finish_reason=response.get(
                "finish_reason"
            ),
            prompt_tokens=int(
                response.get(
                    "prompt_tokens",
                    0,
                )
                or 0
            ),
            completion_tokens=int(
                response.get(
                    "completion_tokens",
                    0,
                )
                or 0
            ),
            total_tokens=int(
                response.get(
                    "total_tokens",
                    0,
                )
                or 0
            ),
            response_id=response.get(
                "response_id"
            ),
            tool_calls=list(
                response.get(
                    "tool_calls",
                    [],
                )
                or []
            ),
            metadata=dict(
                response.get(
                    "metadata",
                    {},
                )
                or {}
            ),
            raw_response=response,
        )

    return LLMResponse(
        content=str(response),
        provider=provider,
        model=model,
        raw_response=response,
    )


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "LLMChunk",
    "ProviderCapabilities",
    "LLMProvider",
    "FunctionLLMProvider",
    "create_message",
    "create_request",
    "normalize_response",
]


