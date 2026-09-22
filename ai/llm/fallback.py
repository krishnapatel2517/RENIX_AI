"""
RENIX AI
LLM Fallback System

Handles reliable fallback between multiple LLM providers.

Responsibilities:
- Provider fallback
- Retry handling
- Failure tracking
- Provider health awareness
- Retry delays
- Error classification
- Synchronous generation
- Asynchronous generation
- Streaming fallback support
"""

from __future__ import annotations

import asyncio
import time

from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from .provider import (
    LLMChunk,
    LLMProvider,
    LLMRequest,
    LLMResponse,
)


# ============================================================================
# TYPES
# ============================================================================

RetryCallback = Callable[
    [str, int, Exception],
    None,
]


# ============================================================================
# FALLBACK CONFIGURATION
# ============================================================================


@dataclass
class FallbackConfig:
    """
    Configuration for the RENIX fallback system.
    """

    enabled: bool = True

    max_provider_attempts: int = 3

    max_retries_per_provider: int = 1

    retry_delay: float = 1.0

    retry_backoff: float = 2.0

    max_retry_delay: float = 15.0

    retry_on_timeout: bool = True

    retry_on_connection_error: bool = True

    retry_on_rate_limit: bool = True

    retry_on_server_error: bool = True

    retry_on_unknown_error: bool = False

    skip_unhealthy_providers: bool = True

    continue_after_partial_stream: bool = False

    notify_retry: bool = True

    def __post_init__(self) -> None:
        self.max_provider_attempts = max(
            1,
            int(
                self.max_provider_attempts
            ),
        )

        self.max_retries_per_provider = max(
            0,
            int(
                self.max_retries_per_provider
            ),
        )

        self.retry_delay = max(
            0.0,
            float(
                self.retry_delay
            ),
        )

        self.retry_backoff = max(
            1.0,
            float(
                self.retry_backoff
            ),
        )

        self.max_retry_delay = max(
            self.retry_delay,
            float(
                self.max_retry_delay
            ),
        )


# ============================================================================
# FAILURE INFORMATION
# ============================================================================


@dataclass
class ProviderFailure:
    """
    Stores information about a provider failure.
    """

    provider: str

    attempt: int

    error: Exception

    timestamp: float = field(
        default_factory=time.time
    )

    retryable: bool = False

    category: str = "unknown"

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "attempt": self.attempt,
            "error": str(self.error),
            "timestamp": self.timestamp,
            "retryable": self.retryable,
            "category": self.category,
        }


# ============================================================================
# FALLBACK RESULT
# ============================================================================


@dataclass
class FallbackResult:
    """
    Result returned by the fallback engine.
    """

    response: Optional[
        LLMResponse
    ] = None

    provider: Optional[str] = None

    attempts: int = 0

    failures: List[
        ProviderFailure
    ] = field(
        default_factory=list
    )

    successful: bool = False

    elapsed_time: float = 0.0

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "attempts": self.attempts,
            "failures": [
                failure.to_dict()
                for failure in self.failures
            ],
            "successful": self.successful,
            "elapsed_time": self.elapsed_time,
        }


# ============================================================================
# ERROR CLASSIFICATION
# ============================================================================


class ErrorClassifier:
    """
    Classifies provider errors so RENIX can decide whether to retry.
    """

    @staticmethod
    def classify(
        error: Exception,
    ) -> str:
        """
        Return a normalized error category.
        """

        name = type(
            error
        ).__name__.lower()

        message = str(
            error
        ).lower()

        if (
            "timeout" in name
            or "timeout" in message
        ):
            return "timeout"

        if (
            "connection" in name
            or "connection" in message
            or "network" in message
        ):
            return "connection"

        if (
            "rate" in message
            or "429" in message
            or "too many requests"
            in message
        ):
            return "rate_limit"

        if (
            "authentication" in message
            or "unauthorized" in message
            or "401" in message
            or "403" in message
        ):
            return "authentication"

        if (
            "server" in message
            or "500" in message
            or "502" in message
            or "503" in message
            or "504" in message
        ):
            return "server"

        if (
            "invalid" in message
            or "bad request" in message
            or "400" in message
        ):
            return "invalid_request"

        return "unknown"

    @classmethod
    def retryable(
        cls,
        error: Exception,
        config: FallbackConfig,
    ) -> bool:
        """
        Determine whether an error should be retried.
        """

        category = cls.classify(
            error
        )

        if category == "timeout":
            return config.retry_on_timeout

        if category == "connection":
            return config.retry_on_connection_error

        if category == "rate_limit":
            return config.retry_on_rate_limit

        if category == "server":
            return config.retry_on_server_error

        if category in {
            "authentication",
            "invalid_request",
        }:
            return False

        return config.retry_on_unknown_error


# ============================================================================
# FALLBACK ENGINE
# ============================================================================


class FallbackEngine:
    """
    Main provider fallback engine.
    """

    def __init__(
        self,
        providers: Optional[
            Sequence[LLMProvider]
        ] = None,
        *,
        config: Optional[
            FallbackConfig
        ] = None,
        on_retry: Optional[
            RetryCallback
        ] = None,
    ) -> None:

        self.config = (
            config
            or FallbackConfig()
        )

        self._providers: List[
            LLMProvider
        ] = list(
            providers or []
        )

        self._failure_history: List[
            ProviderFailure
        ] = []

        self._provider_failures: Dict[
            str,
            int,
        ] = {}

        self._provider_successes: Dict[
            str,
            int,
        ] = {}

        self.on_retry = on_retry

    # ========================================================================
    # PROVIDERS
    # ========================================================================

    def set_providers(
        self,
        providers: Sequence[
            LLMProvider
        ],
    ) -> None:
        """
        Replace the provider list.
        """

        self._providers = list(
            providers
        )

    def add_provider(
        self,
        provider: LLMProvider,
    ) -> None:
        """
        Add a provider if it is not already registered.
        """

        if provider not in self._providers:
            self._providers.append(
                provider
            )

    def remove_provider(
        self,
        provider_name: str,
    ) -> bool:
        """
        Remove a provider by name.
        """

        for index, provider in enumerate(
            self._providers
        ):
            if (
                provider.get_name()
                == provider_name
            ):
                self._providers.pop(
                    index
                )

                return True

        return False

    def get_providers(
        self,
    ) -> List[LLMProvider]:
        """
        Return registered providers.
        """

        return list(
            self._providers
        )

    # ========================================================================
    # PROVIDER FILTERING
    # ========================================================================

    def _available_providers(
        self,
        request: LLMRequest,
    ) -> List[LLMProvider]:
        """
        Return providers capable of handling the request.
        """

        available: List[
            LLMProvider
        ] = []

        for provider in self._providers:

            if (
                self.config.skip_unhealthy_providers
                and not provider.health_check()
            ):
                continue

            if request.model:
                if not provider.supports_model(
                    request.model
                ):
                    continue

            available.append(
                provider
            )

        return available

    # ========================================================================
    # RETRY DELAY
    # ========================================================================

    def _retry_delay(
        self,
        retry_number: int,
    ) -> float:
        """
        Calculate exponential retry delay.
        """

        delay = (
            self.config.retry_delay
            * (
                self.config.retry_backoff
                ** max(
                    0,
                    retry_number - 1,
                )
            )
        )

        return min(
            delay,
            self.config.max_retry_delay,
        )

    def _wait(
        self,
        retry_number: int,
    ) -> None:
        """
        Wait before retrying.
        """

        delay = self._retry_delay(
            retry_number
        )

        if delay > 0:
            time.sleep(
                delay
            )

    async def _wait_async(
        self,
        retry_number: int,
    ) -> None:
        """
        Async retry wait.
        """

        delay = self._retry_delay(
            retry_number
        )

        if delay > 0:
            await asyncio.sleep(
                delay
            )

    # ========================================================================
    # FAILURE TRACKING
    # ========================================================================

    def _record_failure(
        self,
        provider: LLMProvider,
        attempt: int,
        error: Exception,
    ) -> ProviderFailure:
        """
        Record provider failure.
        """

        category = (
            ErrorClassifier.classify(
                error
            )
        )

        retryable = (
            ErrorClassifier.retryable(
                error,
                self.config,
            )
        )

        failure = ProviderFailure(
            provider=provider.get_name(),
            attempt=attempt,
            error=error,
            retryable=retryable,
            category=category,
        )

        self._failure_history.append(
            failure
        )

        name = provider.get_name()

        self._provider_failures[
            name
        ] = (
            self._provider_failures.get(
                name,
                0,
            )
            + 1
        )

        return failure

    def _record_success(
        self,
        provider: LLMProvider,
    ) -> None:
        """
        Record provider success.
        """

        name = provider.get_name()

        self._provider_successes[
            name
        ] = (
            self._provider_successes.get(
                name,
                0,
            )
            + 1
        )

    # ========================================================================
    # SYNCHRONOUS GENERATION
    # ========================================================================

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate a response with automatic provider fallback.
        """

        start = time.perf_counter()

        providers = (
            self._available_providers(
                request
            )
        )

        if not providers:
            raise RuntimeError(
                "No available LLM providers."
            )

        if not self.config.enabled:
            providers = providers[:1]

        providers = providers[
            : self.config.max_provider_attempts
        ]

        failures: List[
            ProviderFailure
        ] = []

        total_attempts = 0

        last_error: Optional[
            Exception
        ] = None

        for provider in providers:

            provider_name = (
                provider.get_name()
            )

            provider_attempts = (
                self.config.max_retries_per_provider
                + 1
            )

            for retry_number in range(
                provider_attempts
            ):

                total_attempts += 1

                try:

                    prepared = (
                        provider.prepare_request(
                            request
                        )
                    )

                    response = (
                        provider.generate(
                            prepared
                        )
                    )

                    self._record_success(
                        provider
                    )

                    return response

                except Exception as exc:

                    last_error = exc

                    failure = (
                        self._record_failure(
                            provider,
                            total_attempts,
                            exc,
                        )
                    )

                    failures.append(
                        failure
                    )

                    if (
                        not failure.retryable
                    ):
                        break

                    if (
                        retry_number
                        >= provider_attempts - 1
                    ):
                        break

                    if self.on_retry:
                        self.on_retry(
                            provider_name,
                            retry_number + 1,
                            exc,
                        )

                    self._wait(
                        retry_number + 1
                    )

            # Continue with next provider.

        elapsed = (
            time.perf_counter()
            - start
        )

        raise RuntimeError(
            self._build_failure_message(
                failures,
                elapsed,
            )
        ) from last_error

    # ========================================================================
    # ASYNC GENERATION
    # ========================================================================

    async def generate_async(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Asynchronously generate a response with fallback.
        """

        start = time.perf_counter()

        providers = (
            self._available_providers(
                request
            )
        )

        if not providers:
            raise RuntimeError(
                "No available LLM providers."
            )

        if not self.config.enabled:
            providers = providers[:1]

        providers = providers[
            : self.config.max_provider_attempts
        ]

        failures: List[
            ProviderFailure
        ] = []

        total_attempts = 0

        last_error: Optional[
            Exception
        ] = None

        for provider in providers:

            provider_name = (
                provider.get_name()
            )

            provider_attempts = (
                self.config.max_retries_per_provider
                + 1
            )

            for retry_number in range(
                provider_attempts
            ):

                total_attempts += 1

                try:

                    prepared = (
                        provider.prepare_request(
                            request
                        )
                    )

                    response = (
                        await provider.generate_async(
                            prepared
                        )
                    )

                    self._record_success(
                        provider
                    )

                    return response

                except Exception as exc:

                    last_error = exc

                    failure = (
                        self._record_failure(
                            provider,
                            total_attempts,
                            exc,
                        )
                    )

                    failures.append(
                        failure
                    )

                    if (
                        not failure.retryable
                    ):
                        break

                    if (
                        retry_number
                        >= provider_attempts - 1
                    ):
                        break

                    if self.on_retry:
                        self.on_retry(
                            provider_name,
                            retry_number + 1,
                            exc,
                        )

                    await self._wait_async(
                        retry_number + 1
                    )

        elapsed = (
            time.perf_counter()
            - start
        )

        raise RuntimeError(
            self._build_failure_message(
                failures,
                elapsed,
            )
        ) from last_error

    # ========================================================================
    # STREAMING
    # ========================================================================

    def generate_stream(
        self,
        request: LLMRequest,
    ) -> Iterator[LLMChunk]:
        """
        Generate a streaming response with fallback.

        Important:
        Once a provider has successfully produced output,
        RENIX does not silently switch providers halfway through
        the response unless explicitly enabled.
        """

        request.stream = True

        providers = (
            self._available_providers(
                request
            )
        )

        if not providers:
            raise RuntimeError(
                "No available streaming providers."
            )

        if not self.config.enabled:
            providers = providers[:1]

        providers = providers[
            : self.config.max_provider_attempts
        ]

        last_error: Optional[
            Exception
        ] = None

        for provider in providers:

            provider_attempts = (
                self.config.max_retries_per_provider
                + 1
            )

            for retry_number in range(
                provider_attempts
            ):

                emitted_output = False

                try:

                    prepared = (
                        provider.prepare_request(
                            request
                        )
                    )

                    for chunk in (
                        provider.generate_stream(
                            prepared
                        )
                    ):

                        if (
                            chunk.content
                        ):
                            emitted_output = True

                        yield chunk

                        if chunk.is_final:
                            self._record_success(
                                provider
                            )

                            return

                    self._record_success(
                        provider
                    )

                    return

                except Exception as exc:

                    last_error = exc

                    failure = (
                        self._record_failure(
                            provider,
                            retry_number + 1,
                            exc,
                        )
                    )

                    if emitted_output:
                        if (
                            not self.config
                            .continue_after_partial_stream
                        ):
                            raise

                    if (
                        not failure.retryable
                    ):
                        break

                    if (
                        retry_number
                        >= provider_attempts - 1
                    ):
                        break

                    if self.on_retry:
                        self.on_retry(
                            provider.get_name(),
                            retry_number + 1,
                            exc,
                        )

                    self._wait(
                        retry_number + 1
                    )

        raise RuntimeError(
            "All streaming providers failed."
        ) from last_error

    async def generate_stream_async(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[LLMChunk]:
        """
        Asynchronously generate a streaming response with fallback.
        """

        request.stream = True

        providers = (
            self._available_providers(
                request
            )
        )

        if not providers:
            raise RuntimeError(
                "No available streaming providers."
            )

        if not self.config.enabled:
            providers = providers[:1]

        providers = providers[
            : self.config.max_provider_attempts
        ]

        last_error: Optional[
            Exception
        ] = None

        for provider in providers:

            provider_attempts = (
                self.config.max_retries_per_provider
                + 1
            )

            for retry_number in range(
                provider_attempts
            ):

                emitted_output = False

                try:

                    prepared = (
                        provider.prepare_request(
                            request
                        )
                    )

                    async for chunk in (
                        provider.generate_stream_async(
                            prepared
                        )
                    ):

                        if (
                            chunk.content
                        ):
                            emitted_output = True

                        yield chunk

                        if chunk.is_final:
                            self._record_success(
                                provider
                            )

                            return

                    self._record_success(
                        provider
                    )

                    return

                except Exception as exc:

                    last_error = exc

                    failure = (
                        self._record_failure(
                            provider,
                            retry_number + 1,
                            exc,
                        )
                    )

                    if emitted_output:
                        if (
                            not self.config
                            .continue_after_partial_stream
                        ):
                            raise

                    if (
                        not failure.retryable
                    ):
                        break

                    if (
                        retry_number
                        >= provider_attempts - 1
                    ):
                        break

                    if self.on_retry:
                        result = self.on_retry(
                            provider.get_name(),
                            retry_number + 1,
                            exc,
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            await result

                    await self._wait_async(
                        retry_number + 1
                    )

        raise RuntimeError(
            "All streaming providers failed."
        ) from last_error

    # ========================================================================
    # FAILURE MESSAGE
    # ========================================================================

    def _build_failure_message(
        self,
        failures: Sequence[
            ProviderFailure
        ],
        elapsed: float,
    ) -> str:
        """
        Build a useful final fallback error.
        """

        if not failures:
            return (
                "LLM fallback failed."
            )

        details = []

        for failure in failures:
            details.append(
                (
                    f"{failure.provider}: "
                    f"{failure.category}: "
                    f"{failure.error}"
                )
            )

        return (
            "All LLM providers failed "
            f"after {len(failures)} failure(s) "
            f"in {elapsed:.2f}s. "
            "Details: "
            + " | ".join(
                details
            )
        )

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def get_failure_history(
        self,
    ) -> List[ProviderFailure]:
        """
        Return a copy of failure history.
        """

        return list(
            self._failure_history
        )

    def clear_failure_history(
        self,
    ) -> None:
        """
        Clear recorded failure history.
        """

        self._failure_history.clear()

    def get_provider_statistics(
        self,
    ) -> Dict[
        str,
        Dict[str, int],
    ]:
        """
        Return success/failure statistics.
        """

        names = set(
            self._provider_failures.keys()
        )

        names.update(
            self._provider_successes.keys()
        )

        return {
            name: {
                "successes": (
                    self._provider_successes.get(
                        name,
                        0,
                    )
                ),
                "failures": (
                    self._provider_failures.get(
                        name,
                        0,
                    )
                ),
            }
            for name in names
        }

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    def update_config(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Update fallback configuration.
        """

        for key, value in kwargs.items():

            if not hasattr(
                self.config,
                key,
            ):
                raise ValueError(
                    f"Unknown fallback option: {key}"
                )

            setattr(
                self.config,
                key,
                value,
            )

        self.config.__post_init__()

    def enable(
        self,
    ) -> None:
        """
        Enable fallback.
        """

        self.config.enabled = True

    def disable(
        self,
    ) -> None:
        """
        Disable fallback.
        """

        self.config.enabled = False

    # ========================================================================
    # STATUS
    # ========================================================================

    def status(
        self,
    ) -> Dict[str, Any]:
        """
        Return fallback engine status.
        """

        return {
            "enabled": self.config.enabled,
            "max_provider_attempts": (
                self.config.max_provider_attempts
            ),
            "max_retries_per_provider": (
                self.config.max_retries_per_provider
            ),
            "retry_delay": (
                self.config.retry_delay
            ),
            "retry_backoff": (
                self.config.retry_backoff
            ),
            "provider_count": len(
                self._providers
            ),
            "providers": [
                provider.get_name()
                for provider in self._providers
            ],
            "statistics": (
                self.get_provider_statistics()
            ),
            "failure_count": len(
                self._failure_history
            ),
        }


# ============================================================================
# DEFAULT FALLBACK ENGINE
# ============================================================================


_default_fallback: Optional[
    FallbackEngine
] = None


def get_default_fallback() -> FallbackEngine:
    """
    Return the global RENIX fallback engine.
    """

    global _default_fallback

    if _default_fallback is None:
        _default_fallback = (
            FallbackEngine()
        )

    return _default_fallback


def set_default_fallback(
    fallback: FallbackEngine,
) -> None:
    """
    Replace the global fallback engine.
    """

    global _default_fallback

    if not isinstance(
        fallback,
        FallbackEngine,
    ):
        raise TypeError(
            "fallback must be a FallbackEngine instance."
        )

    _default_fallback = fallback


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "RetryCallback",
    "FallbackConfig",
    "ProviderFailure",
    "FallbackResult",
    "ErrorClassifier",
    "FallbackEngine",
    "get_default_fallback",
    "set_default_fallback",
]


