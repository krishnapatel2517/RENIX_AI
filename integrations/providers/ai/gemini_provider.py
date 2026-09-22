"""
RENIX Gemini Provider
=====================

Google Gemini implementation of the RENIX AIProvider interface.

This module keeps Gemini-specific API handling isolated from
the rest of RENIX.

Expected environment variable:

    GEMINI_API_KEY

Also supports:

    GOOGLE_API_KEY

Optional:

    RENIX_GEMINI_MODEL
    GEMINI_BASE_URL

The Gemini SDK is imported lazily so RENIX can still start
without the Gemini package installed.
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
# GEMINI PROVIDER
# ============================================================


class GeminiProvider(AIProvider):
    """
    RENIX provider implementation for Google Gemini.

    The provider uses Google's current Python GenAI SDK when
    available.

    Example:

        config = ProviderConfig(
            name="gemini",
            api_key=os.getenv("GEMINI_API_KEY"),
            default_model="gemini-2.5-flash",
        )

        provider = GeminiProvider(config)
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
                name="gemini",
                api_key=(
                    os.getenv("GEMINI_API_KEY")
                    or os.getenv("GOOGLE_API_KEY")
                ),
                base_url=os.getenv(
                    "GEMINI_BASE_URL"
                ),
                default_model=os.getenv(
                    "RENIX_GEMINI_MODEL",
                    "gemini-2.5-flash",
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

        return "gemini"

    # ========================================================
    # CLIENT
    # ========================================================

    def _create_client(self) -> Any:
        """
        Lazily create the Google GenAI client.

        The import is intentionally delayed until the provider
        is actually used.
        """

        try:
            from google import genai
        except ImportError as exc:
            raise ProviderConnectionError(
                "The Google GenAI Python SDK is not installed. "
                "Install the 'google-genai' package."
            ) from exc

        api_key = self.config.api_key

        if not api_key:
            raise ProviderAuthenticationError(
                "Gemini API key is not configured. "
                "Set GEMINI_API_KEY or GOOGLE_API_KEY, "
                "or provide it through RENIX's secrets manager."
            )

        kwargs: dict[str, Any] = {
            "api_key": api_key,
        }

        # Some deployments may provide a custom endpoint.
        # The SDK may not support this parameter in every
        # version, so it is passed only when explicitly set.
        if self.config.base_url:
            kwargs["http_options"] = {
                "base_url": self.config.base_url
            }

        try:
            return genai.Client(**kwargs)

        except TypeError:
            # Compatibility fallback for SDK versions that do
            # not accept the custom HTTP configuration.
            kwargs.pop(
                "http_options",
                None,
            )

            try:
                return genai.Client(
                    **kwargs
                )
            except Exception as exc:
                raise ProviderConnectionError(
                    f"Failed to initialize Gemini client: "
                    f"{exc}"
                ) from exc

        except Exception as exc:
            raise ProviderConnectionError(
                f"Failed to initialize Gemini client: "
                f"{exc}"
            ) from exc

    @property
    def client(self) -> Any:
        """Return lazily initialized Gemini client."""

        if self._client is None:
            self._client = self._create_client()

        return self._client

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def initialize(self) -> None:
        """Initialize Gemini provider."""

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
        Perform a lightweight Gemini availability check.

        No text generation is performed.
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
            # list() is intentionally attempted only when the
            # SDK exposes the models collection.
            models = getattr(
                self.client,
                "models",
                None,
            )

            if models is None:
                self.status = (
                    ProviderStatus.AVAILABLE
                )
                return True

            list_method = getattr(
                models,
                "list",
                None,
            )

            if callable(list_method):
                list_method()

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
    # CONTENT HELPERS
    # ========================================================

    def _build_contents(
        self,
        request: AIRequest,
    ) -> list[Any]:
        """
        Build Gemini contents.

        The current implementation creates a simple user
        content object. System instructions are handled
        separately through generation configuration.
        """

        return [
            request.prompt
        ]

    def _build_config(
        self,
        request: AIRequest,
    ) -> Any:
        """
        Build Gemini GenerateContentConfig.

        The SDK's configuration classes have changed slightly
        between versions, so this method keeps construction
        isolated.
        """

        config_kwargs: dict[str, Any] = {}

        if request.system_prompt:
            config_kwargs[
                "system_instruction"
            ] = request.system_prompt

        if request.temperature is not None:
            config_kwargs[
                "temperature"
            ] = request.temperature

        if request.max_tokens is not None:
            config_kwargs[
                "max_output_tokens"
            ] = request.max_tokens

        if request.top_p is not None:
            config_kwargs[
                "top_p"
            ] = request.top_p

        if request.stop_sequences:
            config_kwargs[
                "stop_sequences"
            ] = list(
                request.stop_sequences
            )

        if request.response_type == ResponseType.JSON:
            config_kwargs[
                "response_mime_type"
            ] = "application/json"

        # Tool definitions are converted when possible.
        if request.tools:
            config_kwargs[
                "tools"
            ] = list(request.tools)

        try:
            from google.genai import types

            return types.GenerateContentConfig(
                **config_kwargs
            )

        except ImportError:
            return config_kwargs

        except TypeError:
            # Keep compatibility with SDK versions whose config
            # class does not accept one of the optional fields.
            safe_config: dict[str, Any] = {}

            supported_fields = {
                "system_instruction",
                "temperature",
                "max_output_tokens",
                "top_p",
                "stop_sequences",
                "response_mime_type",
            }

            for key, value in config_kwargs.items():
                if key in supported_fields:
                    safe_config[key] = value

            try:
                from google.genai import types

                return types.GenerateContentConfig(
                    **safe_config
                )
            except Exception:
                return safe_config

    # ========================================================
    # RESPONSE HELPERS
    # ========================================================

    @staticmethod
    def _extract_text(
        response: Any,
    ) -> str:
        """Extract text from Gemini response."""

        try:
            text = getattr(
                response,
                "text",
                None,
            )

            if text:
                return str(text)

        except Exception:
            pass

        # Compatibility fallback for responses where `.text`
        # is unavailable.
        try:
            candidates = getattr(
                response,
                "candidates",
                None,
            )

            if not candidates:
                return ""

            candidate = candidates[0]

            content = getattr(
                candidate,
                "content",
                None,
            )

            if content is None:
                return ""

            parts = getattr(
                content,
                "parts",
                None,
            )

            if not parts:
                return ""

            texts: list[str] = []

            for part in parts:
                value = getattr(
                    part,
                    "text",
                    None,
                )

                if value:
                    texts.append(
                        str(value)
                    )

            return "".join(texts)

        except Exception:
            return ""

    @staticmethod
    def _extract_usage(
        response: Any,
    ) -> tuple[int, int, int]:
        """Extract token usage metadata."""

        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        if usage is None:
            return 0, 0, 0

        input_tokens = int(
            getattr(
                usage,
                "prompt_token_count",
                0,
            )
            or 0
        )

        output_tokens = int(
            getattr(
                usage,
                "candidates_token_count",
                0,
            )
            or 0
        )

        total_tokens = int(
            getattr(
                usage,
                "total_token_count",
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
    def _extract_finish_reason(
        response: Any,
    ) -> str | None:
        """Extract Gemini finish reason."""

        try:
            candidates = getattr(
                response,
                "candidates",
                None,
            )

            if not candidates:
                return None

            reason = getattr(
                candidates[0],
                "finish_reason",
                None,
            )

            if reason is None:
                return None

            return str(reason)

        except Exception:
            return None

    @staticmethod
    def _extract_request_id(
        response: Any,
    ) -> str | None:
        """Extract a response/request identifier when available."""

        for attribute in (
            "request_id",
            "response_id",
            "id",
        ):
            value = getattr(
                response,
                attribute,
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
        """
        Generate a complete response through Gemini.
        """

        self.validate_request(
            request
        )

        self.record_request()

        started = time.perf_counter()

        model = self.resolve_model(
            request.model
        )

        if not model:
            raise ProviderResponseError(
                "No Gemini model was specified."
            )

        try:
            response = (
                self.client.models.generate_content(
                    model=model,
                    contents=self._build_contents(
                        request
                    ),
                    config=self._build_config(
                        request
                    ),
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

            text = self._extract_text(
                response
            )

            result = AIResponse(
                text=text,
                model=model,
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
                raw_response=response,
                metadata={
                    "provider": "gemini",
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
                f"Gemini request timed out: "
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
        Stream a Gemini response.
        """

        self.validate_request(
            request
        )

        self.record_request()

        self.metrics.streamed_requests += 1

        started = time.perf_counter()

        model = self.resolve_model(
            request.model
        )

        if not model:
            raise ProviderResponseError(
                "No Gemini model was specified."
            )

        try:
            stream = (
                self.client.models.generate_content_stream(
                    model=model,
                    contents=self._build_contents(
                        request
                    ),
                    config=self._build_config(
                        request
                    ),
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

                text = self._extract_text(
                    chunk
                )

                finish_reason = (
                    self._extract_finish_reason(
                        chunk
                    )
                )

                yield AIStreamChunk(
                    text=text,
                    provider=self.provider_name,
                    model=model,
                    request_id=request_id,
                    index=index,
                    is_final=(
                        finish_reason
                        is not None
                    ),
                    finish_reason=finish_reason,
                    metadata={
                        "provider": "gemini"
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
                f"Gemini streaming request timed out: "
                f"{exc}"
            ) from exc

        except Exception as exc:
            self.record_error(exc)

            raise self._translate_error(
                exc
            ) from exc

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    def embed(
        self,
        text: str,
        *,
        model: str | None = None,
    ) -> list[float]:
        """Generate a Gemini embedding."""

        if not text or not str(text).strip():
            raise ValueError(
                "Embedding text cannot be empty."
            )

        self.record_request()

        embedding_model = (
            model
            or self.config.extra.get(
                "embedding_model",
                "gemini-embedding-001",
            )
        )

        try:
            response = (
                self.client.models.embed_content(
                    model=embedding_model,
                    contents=text,
                )
            )

            embeddings = getattr(
                response,
                "embeddings",
                None,
            )

            if not embeddings:
                raise ProviderResponseError(
                    "Gemini returned no embedding data."
                )

            first = embeddings[0]

            values = getattr(
                first,
                "values",
                None,
            )

            if values is None:
                raise ProviderResponseError(
                    "Gemini returned an invalid embedding."
                )

            self.status = (
                ProviderStatus.AVAILABLE
            )

            return [
                float(value)
                for value in values
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
        Translate common Gemini SDK errors into RENIX errors.

        This avoids tightly coupling the rest of RENIX to one
        particular SDK exception hierarchy.
        """

        name = type(error).__name__

        message = str(error)

        lowered = message.lower()

        if (
            "api key" in lowered
            or "authentication" in lowered
            or "unauthenticated" in lowered
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
            or "resource exhausted" in lowered
            or "too many requests" in lowered
        ):
            return ProviderResponseError(
                f"Gemini rate limit reached: "
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


def create_gemini_provider(
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = 60.0,
    max_retries: int = 2,
    client: Any | None = None,
) -> GeminiProvider:
    """
    Create a GeminiProvider using explicit values or
    environment variables.
    """

    resolved_key = (
        api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    config = ProviderConfig(
        name="gemini",
        api_key=resolved_key,
        base_url=(
            base_url
            or os.getenv(
                "GEMINI_BASE_URL"
            )
        ),
        default_model=(
            model
            or os.getenv(
                "RENIX_GEMINI_MODEL",
                "gemini-2.5-flash",
            )
        ),
        timeout=timeout,
        max_retries=max_retries,
    )

    return GeminiProvider(
        config,
        client=client,
    )


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "GeminiProvider",
    "create_gemini_provider",
]


