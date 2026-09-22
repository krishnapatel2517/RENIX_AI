"""
RENIX AI
LLM Router

Central routing layer for selecting and using the best available
LLM provider/model.

The router supports:

- Multiple providers
- Provider registration
- Provider priorities
- Model selection
- Capability-based routing
- Automatic fallback
- Health-aware routing
- Request metadata
- Synchronous generation
- Asynchronous generation
- Streaming
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
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
    ProviderCapabilities,
)


# ============================================================================
# ROUTING DATA
# ============================================================================


@dataclass
class ProviderRegistration:
    """
    Stores information about a registered LLM provider.
    """

    provider: LLMProvider

    priority: int = 100

    enabled: bool = True

    weight: int = 1

    tags: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    failures: int = 0

    successes: int = 0

    last_error: Optional[str] = None

    def is_available(self) -> bool:
        """
        Determine whether this provider can currently be used.
        """

        if not self.enabled:
            return False

        return self.provider.health_check()

    def score(
        self,
        request: LLMRequest,
    ) -> float:
        """
        Calculate a routing score.

        Higher scores are preferred.
        """

        score = float(
            self.priority
        )

        capabilities = (
            self.provider.get_capabilities()
        )

        if request.stream:
            if capabilities.streaming:
                score += 50
            else:
                score -= 100

        if request.tools:
            if capabilities.tool_calling:
                score += 40
            else:
                score -= 100

        if request.response_format:
            if capabilities.structured_output:
                score += 25
            else:
                score -= 50

        if request.metadata.get(
            "requires_vision",
            False,
        ):
            if capabilities.vision:
                score += 50
            else:
                score -= 100

        if request.metadata.get(
            "requires_reasoning",
            False,
        ):
            if capabilities.reasoning:
                score += 50
            else:
                score -= 75

        score += (
            self.successes * 0.1
        )

        score -= (
            self.failures * 5
        )

        score += (
            max(
                1,
                self.weight,
            )
            * 0.01
        )

        return score


# ============================================================================
# ROUTER
# ============================================================================


class LLMRouter:
    """
    Main LLM routing system for RENIX.
    """

    def __init__(
        self,
        providers: Optional[
            Sequence[LLMProvider]
        ] = None,
        *,
        default_provider: Optional[str] = None,
        fallback_enabled: bool = True,
        max_fallbacks: int = 3,
    ) -> None:

        self._providers: Dict[
            str,
            ProviderRegistration,
        ] = {}

        self.default_provider = (
            default_provider
        )

        self.fallback_enabled = (
            fallback_enabled
        )

        self.max_fallbacks = max(
            0,
            max_fallbacks,
        )

        if providers:
            for provider in providers:
                self.register(
                    provider
                )

    # ========================================================================
    # PROVIDER REGISTRATION
    # ========================================================================

    def register(
        self,
        provider: LLMProvider,
        *,
        priority: int = 100,
        enabled: bool = True,
        weight: int = 1,
        tags: Optional[
            Sequence[str]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        replace: bool = False,
    ) -> None:
        """
        Register an LLM provider.
        """

        name = provider.get_name()

        if (
            name in self._providers
            and not replace
        ):
            raise ValueError(
                f"Provider '{name}' is already registered."
            )

        self._providers[name] = (
            ProviderRegistration(
                provider=provider,
                priority=priority,
                enabled=enabled,
                weight=max(
                    1,
                    weight,
                ),
                tags=list(
                    tags or []
                ),
                metadata=dict(
                    metadata or {}
                ),
            )
        )

        if self.default_provider is None:
            self.default_provider = name

    def unregister(
        self,
        provider_name: str,
    ) -> bool:
        """
        Remove a provider from the router.
        """

        if (
            provider_name
            not in self._providers
        ):
            return False

        del self._providers[
            provider_name
        ]

        if (
            self.default_provider
            == provider_name
        ):
            self.default_provider = (
                next(
                    iter(
                        self._providers
                    ),
                    None,
                )
            )

        return True

    def enable(
        self,
        provider_name: str,
    ) -> None:
        """
        Enable a provider.
        """

        registration = self._get_registration(
            provider_name
        )

        registration.enabled = True

    def disable(
        self,
        provider_name: str,
    ) -> None:
        """
        Disable a provider.
        """

        registration = self._get_registration(
            provider_name
        )

        registration.enabled = False

    # ========================================================================
    # PROVIDER LOOKUP
    # ========================================================================

    def _get_registration(
        self,
        provider_name: str,
    ) -> ProviderRegistration:
        """
        Get a provider registration.
        """

        try:
            return self._providers[
                provider_name
            ]
        except KeyError as exc:
            raise ValueError(
                f"Provider '{provider_name}' "
                f"is not registered."
            ) from exc

    def get_provider(
        self,
        provider_name: str,
    ) -> LLMProvider:
        """
        Get a provider instance by name.
        """

        return self._get_registration(
            provider_name
        ).provider

    def get_providers(
        self,
        *,
        enabled_only: bool = False,
    ) -> List[LLMProvider]:
        """
        Return registered providers.
        """

        registrations = list(
            self._providers.values()
        )

        if enabled_only:
            registrations = [
                registration
                for registration in registrations
                if registration.enabled
            ]

        return [
            registration.provider
            for registration in registrations
        ]

    def provider_names(
        self,
    ) -> List[str]:
        """
        Return all registered provider names.
        """

        return list(
            self._providers.keys()
        )

    # ========================================================================
    # MODEL LOOKUP
    # ========================================================================

    def find_provider_for_model(
        self,
        model: str,
    ) -> Optional[LLMProvider]:
        """
        Find the first available provider supporting a model.
        """

        candidates: List[
            ProviderRegistration
        ] = []

        for registration in (
            self._providers.values()
        ):
            if not registration.enabled:
                continue

            if not registration.provider.supports_model(
                model
            ):
                continue

            candidates.append(
                registration
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item.priority,
            reverse=True,
        )

        return candidates[0].provider

    # ========================================================================
    # CAPABILITY ROUTING
    # ========================================================================

    def find_providers_by_capability(
        self,
        capability: str,
    ) -> List[LLMProvider]:
        """
        Find enabled providers supporting a capability.
        """

        candidates: List[
            ProviderRegistration
        ] = []

        for registration in (
            self._providers.values()
        ):
            if not registration.enabled:
                continue

            if registration.provider.supports(
                capability
            ):
                candidates.append(
                    registration
                )

        candidates.sort(
            key=lambda item: item.priority,
            reverse=True,
        )

        return [
            item.provider
            for item in candidates
        ]

    # ========================================================================
    # ROUTING
    # ========================================================================

    def _candidate_registrations(
        self,
        request: LLMRequest,
        *,
        preferred_provider: Optional[str] = None,
    ) -> List[ProviderRegistration]:
        """
        Build an ordered list of candidate providers.
        """

        candidates: List[
            ProviderRegistration
        ] = []

        for registration in (
            self._providers.values()
        ):
            if not registration.enabled:
                continue

            provider = registration.provider

            if not provider.health_check():
                continue

            if request.model:
                if not provider.supports_model(
                    request.model
                ):
                    continue

            candidates.append(
                registration
            )

        if preferred_provider:
            preferred = [
                registration
                for registration in candidates
                if registration.provider.get_name()
                == preferred_provider
            ]

            remaining = [
                registration
                for registration in candidates
                if registration.provider.get_name()
                != preferred_provider
            ]

            candidates = (
                preferred
                + remaining
            )

        candidates.sort(
            key=lambda item: item.score(
                request
            ),
            reverse=True,
        )

        return candidates

    def select_provider(
        self,
        request: LLMRequest,
        *,
        preferred_provider: Optional[str] = None,
    ) -> LLMProvider:
        """
        Select the best available provider for a request.
        """

        candidates = (
            self._candidate_registrations(
                request,
                preferred_provider=preferred_provider,
            )
        )

        if not candidates:
            raise RuntimeError(
                "No suitable LLM provider is available."
            )

        return candidates[0].provider

    # ========================================================================
    # EXECUTION
    # ========================================================================

    def generate(
        self,
        request: LLMRequest,
        *,
        provider: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a complete response using the best provider.
        """

        if not request.messages:
            raise ValueError(
                "LLM request must contain at least one message."
            )

        candidates = (
            self._candidate_registrations(
                request,
                preferred_provider=provider
                or self.default_provider,
            )
        )

        if not candidates:
            raise RuntimeError(
                "No available LLM provider can handle this request."
            )

        if not self.fallback_enabled:
            candidates = candidates[:1]
        else:
            candidates = candidates[
                : self.max_fallbacks + 1
            ]

        errors: List[
            Tuple[str, Exception]
        ] = []

        for registration in candidates:
            provider_instance = (
                registration.provider
            )

            try:
                prepared = (
                    provider_instance.prepare_request(
                        request
                    )
                )

                response = (
                    provider_instance.generate(
                        prepared
                    )
                )

                registration.successes += 1

                registration.last_error = None

                return response

            except Exception as exc:
                registration.failures += 1

                registration.last_error = (
                    str(exc)
                )

                errors.append(
                    (
                        provider_instance.get_name(),
                        exc,
                    )
                )

                if not self.fallback_enabled:
                    break

        message = (
            "All available LLM providers failed."
        )

        if errors:
            details = "; ".join(
                f"{name}: {error}"
                for name, error in errors
            )

            message = (
                f"{message} "
                f"Failures: {details}"
            )

        raise RuntimeError(
            message
        ) from (
            errors[-1][1]
            if errors
            else None
        )

    async def generate_async(
        self,
        request: LLMRequest,
        *,
        provider: Optional[str] = None,
    ) -> LLMResponse:
        """
        Asynchronously generate a complete response.
        """

        candidates = (
            self._candidate_registrations(
                request,
                preferred_provider=provider
                or self.default_provider,
            )
        )

        if not candidates:
            raise RuntimeError(
                "No available LLM provider can handle this request."
            )

        if not self.fallback_enabled:
            candidates = candidates[:1]
        else:
            candidates = candidates[
                : self.max_fallbacks + 1
            ]

        errors: List[
            Tuple[str, Exception]
        ] = []

        for registration in candidates:
            provider_instance = (
                registration.provider
            )

            try:
                prepared = (
                    provider_instance.prepare_request(
                        request
                    )
                )

                response = (
                    await provider_instance.generate_async(
                        prepared
                    )
                )

                registration.successes += 1

                registration.last_error = None

                return response

            except Exception as exc:
                registration.failures += 1

                registration.last_error = (
                    str(exc)
                )

                errors.append(
                    (
                        provider_instance.get_name(),
                        exc,
                    )
                )

        message = (
            "All available LLM providers failed."
        )

        if errors:
            message += (
                " Failures: "
                + "; ".join(
                    f"{name}: {error}"
                    for name, error in errors
                )
            )

        raise RuntimeError(
            message
        ) from (
            errors[-1][1]
            if errors
            else None
        )

    # ========================================================================
    # STREAMING
    # ========================================================================

    def generate_stream(
        self,
        request: LLMRequest,
        *,
        provider: Optional[str] = None,
    ) -> Iterator[LLMChunk]:
        """
        Generate a streaming response.
        """

        request.stream = True

        candidates = (
            self._candidate_registrations(
                request,
                preferred_provider=provider
                or self.default_provider,
            )
        )

        if not candidates:
            raise RuntimeError(
                "No streaming-capable LLM provider is available."
            )

        if not self.fallback_enabled:
            candidates = candidates[:1]
        else:
            candidates = candidates[
                : self.max_fallbacks + 1
            ]

        last_error: Optional[
            Exception
        ] = None

        for registration in candidates:
            provider_instance = (
                registration.provider
            )

            if not provider_instance.supports(
                "streaming"
            ):
                continue

            try:
                prepared = (
                    provider_instance.prepare_request(
                        request
                    )
                )

                had_output = False

                for chunk in (
                    provider_instance.generate_stream(
                        prepared
                    )
                ):
                    had_output = True

                    yield chunk

                registration.successes += 1

                registration.last_error = None

                return

            except Exception as exc:
                registration.failures += 1

                registration.last_error = (
                    str(exc)
                )

                last_error = exc

                if had_output:
                    raise

        raise RuntimeError(
            "All streaming LLM providers failed."
        ) from last_error

    async def generate_stream_async(
        self,
        request: LLMRequest,
        *,
        provider: Optional[str] = None,
    ) -> AsyncIterator[LLMChunk]:
        """
        Asynchronously generate a streaming response.
        """

        request.stream = True

        candidates = (
            self._candidate_registrations(
                request,
                preferred_provider=provider
                or self.default_provider,
            )
        )

        if not candidates:
            raise RuntimeError(
                "No streaming-capable LLM provider is available."
            )

        if not self.fallback_enabled:
            candidates = candidates[:1]
        else:
            candidates = candidates[
                : self.max_fallbacks + 1
            ]

        last_error: Optional[
            Exception
        ] = None

        for registration in candidates:
            provider_instance = (
                registration.provider
            )

            if not provider_instance.supports(
                "streaming"
            ):
                continue

            try:
                prepared = (
                    provider_instance.prepare_request(
                        request
                    )
                )

                async for chunk in (
                    provider_instance.generate_stream_async(
                        prepared
                    )
                ):
                    yield chunk

                registration.successes += 1

                registration.last_error = None

                return

            except Exception as exc:
                registration.failures += 1

                registration.last_error = (
                    str(exc)
                )

                last_error = exc

        raise RuntimeError(
            "All streaming LLM providers failed."
        ) from last_error

    # ========================================================================
    # SIMPLE CHAT
    # ========================================================================

    def chat(
        self,
        message: str,
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        provider: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> LLMResponse:
        """
        Simple text chat interface.
        """

        request = LLMRequest(
            messages=[
                self._message(
                    "user",
                    message,
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
            request,
            provider=provider,
        )

    async def chat_async(
        self,
        message: str,
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        provider: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> LLMResponse:
        """
        Asynchronous simple text chat interface.
        """

        request = LLMRequest(
            messages=[
                self._message(
                    "user",
                    message,
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
            request,
            provider=provider,
        )

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    @staticmethod
    def _message(
        role: str,
        content: str,
    ) -> Any:
        """
        Create an LLM message without exposing message construction
        throughout the router.
        """

        from .provider import LLMMessage

        return LLMMessage(
            role=role,
            content=content,
        )

    # ========================================================================
    # STATUS / HEALTH
    # ========================================================================

    def health_check(
        self,
    ) -> Dict[str, bool]:
        """
        Return health status for every registered provider.
        """

        return {
            name: registration.provider.health_check()
            for name, registration
            in self._providers.items()
        }

    def status(
        self,
    ) -> Dict[str, Any]:
        """
        Return complete router status.
        """

        providers: Dict[
            str,
            Any,
        ] = {}

        for name, registration in (
            self._providers.items()
        ):
            providers[name] = {
                "enabled": registration.enabled,
                "priority": registration.priority,
                "weight": registration.weight,
                "tags": list(
                    registration.tags
                ),
                "successes": registration.successes,
                "failures": registration.failures,
                "last_error": registration.last_error,
                "provider": registration.provider.status(),
            }

        return {
            "default_provider": self.default_provider,
            "fallback_enabled": self.fallback_enabled,
            "max_fallbacks": self.max_fallbacks,
            "provider_count": len(
                self._providers
            ),
            "providers": providers,
        }

    # ========================================================================
    # ROUTER CONFIGURATION
    # ========================================================================

    def set_default_provider(
        self,
        provider_name: str,
    ) -> None:
        """
        Set the default provider.
        """

        self._get_registration(
            provider_name
        )

        self.default_provider = (
            provider_name
        )

    def set_fallback(
        self,
        enabled: bool,
    ) -> None:
        """
        Enable or disable provider fallback.
        """

        self.fallback_enabled = bool(
            enabled
        )

    def set_max_fallbacks(
        self,
        count: int,
    ) -> None:
        """
        Set maximum fallback attempts.
        """

        if count < 0:
            raise ValueError(
                "Fallback count cannot be negative."
            )

        self.max_fallbacks = int(
            count
        )


# ============================================================================
# DEFAULT ROUTER
# ============================================================================


_default_router: Optional[
    LLMRouter
] = None


def get_default_router() -> LLMRouter:
    """
    Return the global RENIX LLM router.

    The router is created lazily so importing this module does
    not initialize external providers automatically.
    """

    global _default_router

    if _default_router is None:
        _default_router = LLMRouter()

    return _default_router


def set_default_router(
    router: LLMRouter,
) -> None:
    """
    Replace the global RENIX LLM router.
    """

    global _default_router

    if not isinstance(
        router,
        LLMRouter,
    ):
        raise TypeError(
            "router must be an LLMRouter instance."
        )

    _default_router = router


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "ProviderRegistration",
    "LLMRouter",
    "get_default_router",
    "set_default_router",
]


