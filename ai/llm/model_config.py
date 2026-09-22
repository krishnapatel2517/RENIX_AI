"""
RENIX AI
LLM Model Configuration

Centralized configuration for LLM models and providers.

Responsibilities:
- Model definitions
- Provider/model capabilities
- Generation parameters
- Context limits
- Token limits
- Temperature configuration
- Model aliases
- Default model selection
- Model routing preferences
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
    asdict,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================


@dataclass
class ModelConfig:
    """
    Configuration describing one LLM model.
    """

    name: str

    provider: str

    display_name: Optional[str] = None

    description: str = ""

    context_window: int = 8192

    max_output_tokens: int = 4096

    temperature: float = 0.7

    top_p: float = 1.0

    top_k: Optional[int] = None

    frequency_penalty: float = 0.0

    presence_penalty: float = 0.0

    repetition_penalty: Optional[float] = None

    supports_streaming: bool = True

    supports_vision: bool = False

    supports_tools: bool = False

    supports_function_calling: bool = False

    supports_json: bool = False

    supports_audio: bool = False

    supports_embeddings: bool = False

    supports_reasoning: bool = False

    supports_system_messages: bool = True

    supports_multimodal: bool = False

    languages: List[str] = field(
        default_factory=lambda: [
            "en"
        ]
    )

    aliases: List[str] = field(
        default_factory=list
    )

    tags: List[str] = field(
        default_factory=list
    )

    priority: int = 100

    enabled: bool = True

    cost_input_per_million: float = 0.0

    cost_output_per_million: float = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __post_init__(self) -> None:
        self.name = str(
            self.name
        ).strip()

        self.provider = str(
            self.provider
        ).strip()

        if not self.display_name:
            self.display_name = self.name

        self.context_window = max(
            1,
            int(
                self.context_window
            ),
        )

        self.max_output_tokens = max(
            1,
            int(
                self.max_output_tokens
            ),
        )

        self.temperature = min(
            2.0,
            max(
                0.0,
                float(
                    self.temperature
                ),
            ),
        )

        self.top_p = min(
            1.0,
            max(
                0.0,
                float(
                    self.top_p
                ),
            ),
        )

        if self.top_k is not None:
            self.top_k = max(
                1,
                int(
                    self.top_k
                ),
            )

        self.priority = int(
            self.priority
        )

    # ========================================================================
    # ALIASES
    # ========================================================================

    def matches(
        self,
        model_name: str,
    ) -> bool:
        """
        Check whether a model name matches this configuration.
        """

        if not model_name:
            return False

        normalized = (
            model_name.strip().lower()
        )

        candidates = {
            self.name.lower(),
            self.display_name.lower()
            if self.display_name
            else "",
        }

        candidates.update(
            alias.lower()
            for alias in self.aliases
        )

        return normalized in candidates

    # ========================================================================
    # CAPABILITIES
    # ========================================================================

    def supports(
        self,
        capability: str,
    ) -> bool:
        """
        Check whether this model supports a capability.
        """

        capability = (
            capability.strip()
            .lower()
        )

        mapping = {
            "streaming": self.supports_streaming,
            "vision": self.supports_vision,
            "tools": self.supports_tools,
            "function_calling": (
                self.supports_function_calling
            ),
            "json": self.supports_json,
            "audio": self.supports_audio,
            "embeddings": (
                self.supports_embeddings
            ),
            "reasoning": (
                self.supports_reasoning
            ),
            "system": (
                self.supports_system_messages
            ),
            "multimodal": (
                self.supports_multimodal
            ),
        }

        return bool(
            mapping.get(
                capability,
                False,
            )
        )

    def capability_list(
        self,
    ) -> List[str]:
        """
        Return all supported capabilities.
        """

        capabilities = [
            "streaming",
            "vision",
            "tools",
            "function_calling",
            "json",
            "audio",
            "embeddings",
            "reasoning",
            "system",
            "multimodal",
        ]

        return [
            capability
            for capability in capabilities
            if self.supports(
                capability
            )
        ]

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Convert configuration into a dictionary.
        """

        return asdict(
            self
        )

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "ModelConfig":
        """
        Create configuration from a dictionary.
        """

        return cls(
            **data
        )


# ============================================================================
# GENERATION CONFIGURATION
# ============================================================================


@dataclass
class GenerationConfig:
    """
    Runtime text-generation parameters.
    """

    temperature: float = 0.7

    top_p: float = 1.0

    top_k: Optional[int] = None

    max_tokens: int = 4096

    stop_sequences: List[str] = field(
        default_factory=list
    )

    frequency_penalty: float = 0.0

    presence_penalty: float = 0.0

    repetition_penalty: Optional[float] = None

    seed: Optional[int] = None

    stream: bool = False

    response_format: Optional[
        str
    ] = None

    reasoning_effort: Optional[
        str
    ] = None

    def __post_init__(self) -> None:
        self.temperature = min(
            2.0,
            max(
                0.0,
                float(
                    self.temperature
                ),
            ),
        )

        self.top_p = min(
            1.0,
            max(
                0.0,
                float(
                    self.top_p
                ),
            ),
        )

        self.max_tokens = max(
            1,
            int(
                self.max_tokens
            ),
        )

        if self.top_k is not None:
            self.top_k = max(
                1,
                int(
                    self.top_k
                ),
            )

        if self.seed is not None:
            self.seed = int(
                self.seed
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return asdict(
            self
        )

    @classmethod
    def from_model(
        cls,
        model: ModelConfig,
    ) -> "GenerationConfig":
        """
        Create generation settings using model defaults.
        """

        return cls(
            temperature=model.temperature,
            top_p=model.top_p,
            top_k=model.top_k,
            max_tokens=model.max_output_tokens,
            frequency_penalty=(
                model.frequency_penalty
            ),
            presence_penalty=(
                model.presence_penalty
            ),
            repetition_penalty=(
                model.repetition_penalty
            ),
        )


# ============================================================================
# MODEL REGISTRY
# ============================================================================


class ModelRegistry:
    """
    Registry containing all RENIX model configurations.
    """

    def __init__(
        self,
    ) -> None:

        self._models: Dict[
            str,
            ModelConfig,
        ] = {}

        self._aliases: Dict[
            str,
            str,
        ] = {}

        self._register_builtin_models()

    # ========================================================================
    # REGISTRATION
    # ========================================================================

    def register(
        self,
        model: ModelConfig,
    ) -> None:
        """
        Register a model.
        """

        if not isinstance(
            model,
            ModelConfig,
        ):
            raise TypeError(
                "model must be a ModelConfig instance."
            )

        key = model.name.lower()

        self._models[key] = model

        for alias in model.aliases:
            self._aliases[
                alias.lower()
            ] = key

    def unregister(
        self,
        model_name: str,
    ) -> bool:
        """
        Remove a model from the registry.
        """

        key = self._resolve_key(
            model_name
        )

        if key is None:
            return False

        model = self._models.pop(
            key
        )

        aliases_to_remove = [
            alias
            for alias, target in (
                self._aliases.items()
            )
            if target == key
        ]

        for alias in aliases_to_remove:
            self._aliases.pop(
                alias,
                None,
            )

        return model is not None

    # ========================================================================
    # LOOKUP
    # ========================================================================

    def _resolve_key(
        self,
        model_name: str,
    ) -> Optional[str]:
        """
        Resolve a model name or alias.
        """

        if not model_name:
            return None

        normalized = (
            model_name.strip()
            .lower()
        )

        if normalized in self._models:
            return normalized

        return self._aliases.get(
            normalized
        )

    def get(
        self,
        model_name: str,
    ) -> Optional[ModelConfig]:
        """
        Get a model configuration.
        """

        key = self._resolve_key(
            model_name
        )

        if key is None:
            return None

        return self._models.get(
            key
        )

    def require(
        self,
        model_name: str,
    ) -> ModelConfig:
        """
        Get a model or raise an error.
        """

        model = self.get(
            model_name
        )

        if model is None:
            raise KeyError(
                f"Unknown LLM model: {model_name}"
            )

        return model

    def has(
        self,
        model_name: str,
    ) -> bool:
        """
        Check whether a model exists.
        """

        return (
            self.get(
                model_name
            )
            is not None
        )

    # ========================================================================
    # LISTING
    # ========================================================================

    def all(
        self,
        *,
        enabled_only: bool = False,
    ) -> List[ModelConfig]:
        """
        Return registered models.
        """

        models = list(
            self._models.values()
        )

        if enabled_only:
            models = [
                model
                for model in models
                if model.enabled
            ]

        return sorted(
            models,
            key=lambda model: (
                model.priority,
                model.name,
            ),
        )

    def providers(
        self,
    ) -> List[str]:
        """
        Return unique provider names.
        """

        return sorted(
            {
                model.provider
                for model in self._models.values()
            }
        )

    def by_provider(
        self,
        provider: str,
    ) -> List[ModelConfig]:
        """
        Return models belonging to a provider.
        """

        normalized = (
            provider.strip().lower()
        )

        return [
            model
            for model in self.all(
                enabled_only=True
            )
            if model.provider.lower()
            == normalized
        ]

    # ========================================================================
    # CAPABILITY SEARCH
    # ========================================================================

    def find_by_capability(
        self,
        capability: str,
    ) -> List[ModelConfig]:
        """
        Find models supporting a capability.
        """

        return [
            model
            for model in self.all(
                enabled_only=True
            )
            if model.supports(
                capability
            )
        ]

    def find(
        self,
        *,
        provider: Optional[str] = None,
        capability: Optional[str] = None,
        tags: Optional[
            List[str]
        ] = None,
    ) -> List[ModelConfig]:
        """
        Flexible model search.
        """

        models = self.all(
            enabled_only=True
        )

        if provider:
            normalized_provider = (
                provider.lower()
            )

            models = [
                model
                for model in models
                if model.provider.lower()
                == normalized_provider
            ]

        if capability:
            models = [
                model
                for model in models
                if model.supports(
                    capability
                )
            ]

        if tags:
            wanted_tags = {
                tag.lower()
                for tag in tags
            }

            models = [
                model
                for model in models
                if wanted_tags.issubset(
                    {
                        tag.lower()
                        for tag in model.tags
                    }
                )
            ]

        return models

    # ========================================================================
    # BUILT-IN MODELS
    # ========================================================================

    def _register_builtin_models(
        self,
    ) -> None:
        """
        Register generic RENIX model profiles.

        Provider-specific availability is intentionally kept
        configurable instead of hard-coding API credentials.
        """

        self.register(
            ModelConfig(
                name="default",
                provider="auto",
                display_name="RENIX Default",
                description=(
                    "RENIX automatically selected model."
                ),
                aliases=[
                    "renix",
                    "auto",
                ],
                tags=[
                    "general",
                    "default",
                ],
                supports_streaming=True,
                supports_tools=True,
                supports_function_calling=True,
                supports_json=True,
                supports_reasoning=True,
                supports_multimodal=True,
                priority=1,
            )
        )

        self.register(
            ModelConfig(
                name="fast",
                provider="auto",
                display_name="RENIX Fast",
                description=(
                    "Fast model profile for everyday commands."
                ),
                aliases=[
                    "quick",
                    "speed",
                ],
                tags=[
                    "fast",
                    "general",
                ],
                temperature=0.5,
                max_output_tokens=2048,
                supports_streaming=True,
                supports_tools=True,
                supports_function_calling=True,
                supports_json=True,
                priority=10,
            )
        )

        self.register(
            ModelConfig(
                name="reasoning",
                provider="auto",
                display_name="RENIX Reasoning",
                description=(
                    "Reasoning-focused model profile."
                ),
                aliases=[
                    "think",
                    "deep",
                ],
                temperature=0.3,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_function_calling=True,
                supports_json=True,
                supports_reasoning=True,
                priority=20,
            )
        )

        self.register(
            ModelConfig(
                name="vision",
                provider="auto",
                display_name="RENIX Vision",
                description=(
                    "Multimodal model profile."
                ),
                aliases=[
                    "visual",
                    "multimodal",
                ],
                supports_streaming=True,
                supports_vision=True,
                supports_multimodal=True,
                supports_tools=True,
                supports_function_calling=True,
                supports_json=True,
                priority=30,
            )
        )

        self.register(
            ModelConfig(
                name="coding",
                provider="auto",
                display_name="RENIX Coding",
                description=(
                    "Coding-focused model profile."
                ),
                aliases=[
                    "code",
                    "developer",
                    "programming",
                ],
                temperature=0.2,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_function_calling=True,
                supports_json=True,
                supports_reasoning=True,
                priority=40,
            )
        )

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def export(
        self,
    ) -> Dict[str, Any]:
        """
        Export the complete registry.
        """

        return {
            "models": [
                model.to_dict()
                for model in self.all()
            ]
        }

    def import_models(
        self,
        data: Dict[str, Any],
    ) -> None:
        """
        Import models from a dictionary.
        """

        models = data.get(
            "models",
            [],
        )

        if not isinstance(
            models,
            list,
        ):
            raise ValueError(
                "'models' must be a list."
            )

        for item in models:

            if not isinstance(
                item,
                dict,
            ):
                continue

            self.register(
                ModelConfig.from_dict(
                    item
                )
            )


# ============================================================================
# ROUTING PROFILE
# ============================================================================


@dataclass
class ModelRoutingProfile:
    """
    Defines which model profile should be used for a task.
    """

    name: str

    primary_model: str

    fallback_models: List[str] = field(
        default_factory=list
    )

    temperature: Optional[
        float
    ] = None

    max_tokens: Optional[
        int
    ] = None

    require_capabilities: List[
        str
    ] = field(
        default_factory=list
    )

    prefer_fast: bool = False

    prefer_reasoning: bool = False

    prefer_vision: bool = False

    prefer_coding: bool = False

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return asdict(
            self
        )


# ============================================================================
# DEFAULT ROUTING PROFILES
# ============================================================================


def create_default_routing_profiles(
) -> Dict[
    str,
    ModelRoutingProfile,
]:
    """
    Create RENIX's default task routing profiles.
    """

    return {
        "general": ModelRoutingProfile(
            name="general",
            primary_model="default",
            fallback_models=[
                "fast",
                "reasoning",
            ],
        ),
        "fast": ModelRoutingProfile(
            name="fast",
            primary_model="fast",
            fallback_models=[
                "default",
            ],
            prefer_fast=True,
        ),
        "reasoning": ModelRoutingProfile(
            name="reasoning",
            primary_model="reasoning",
            fallback_models=[
                "default",
                "fast",
            ],
            prefer_reasoning=True,
        ),
        "coding": ModelRoutingProfile(
            name="coding",
            primary_model="coding",
            fallback_models=[
                "reasoning",
                "default",
            ],
            require_capabilities=[
                "tools",
            ],
            prefer_coding=True,
        ),
        "vision": ModelRoutingProfile(
            name="vision",
            primary_model="vision",
            fallback_models=[
                "default",
            ],
            require_capabilities=[
                "vision",
            ],
            prefer_vision=True,
        ),
        "research": ModelRoutingProfile(
            name="research",
            primary_model="reasoning",
            fallback_models=[
                "default",
                "fast",
            ],
            prefer_reasoning=True,
        ),
        "conversation": ModelRoutingProfile(
            name="conversation",
            primary_model="default",
            fallback_models=[
                "fast",
            ],
        ),
    }


# ============================================================================
# GLOBAL REGISTRY
# ============================================================================


_default_registry: Optional[
    ModelRegistry
] = None


def get_model_registry() -> ModelRegistry:
    """
    Return the global RENIX model registry.
    """

    global _default_registry

    if _default_registry is None:
        _default_registry = (
            ModelRegistry()
        )

    return _default_registry


def set_model_registry(
    registry: ModelRegistry,
) -> None:
    """
    Replace the global model registry.
    """

    global _default_registry

    if not isinstance(
        registry,
        ModelRegistry,
    ):
        raise TypeError(
            "registry must be a ModelRegistry instance."
        )

    _default_registry = registry


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def get_model(
    model_name: str,
) -> Optional[ModelConfig]:
    """
    Get a model from the global registry.
    """

    return get_model_registry().get(
        model_name
    )


def require_model(
    model_name: str,
) -> ModelConfig:
    """
    Require a model from the global registry.
    """

    return get_model_registry().require(
        model_name
    )


def list_models(
    *,
    enabled_only: bool = True,
) -> List[ModelConfig]:
    """
    List registered models.
    """

    return get_model_registry().all(
        enabled_only=enabled_only
    )


def find_models(
    *,
    provider: Optional[str] = None,
    capability: Optional[str] = None,
    tags: Optional[
        List[str]
    ] = None,
) -> List[ModelConfig]:
    """
    Find models using the global registry.
    """

    return get_model_registry().find(
        provider=provider,
        capability=capability,
        tags=tags,
    )


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "ModelConfig",
    "GenerationConfig",
    "ModelRegistry",
    "ModelRoutingProfile",
    "create_default_routing_profiles",
    "get_model_registry",
    "set_model_registry",
    "get_model",
    "require_model",
    "list_models",
    "find_models",
]


