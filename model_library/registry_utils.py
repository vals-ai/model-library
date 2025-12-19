from functools import cache
from pathlib import Path
from typing import TypedDict

import tiktoken

from model_library.base import LLM, LLMConfig, ProviderConfig
from model_library.base.output import QueryResultCost, QueryResultMetadata
from model_library.register_models import (
    CostProperties,
    ModelConfig,
    get_model_registry,
    get_provider_registry,
)

ALL_MODELS_PATH = Path(__file__).parent / "config" / "all_models.json"


def create_config(
    registry_config: ModelConfig, override_config: LLMConfig | None
) -> LLMConfig:
    """
    Converts a model registry entry to the necessary LLMConfig.
    May optionally override the config with a provided override_config,
    only set fields will be used to override.
    """
    config: object = {}

    properties = registry_config.properties
    supports = registry_config.supports
    provider_properties = registry_config.provider_properties
    defaults = registry_config.default_parameters

    if properties:
        if properties.max_tokens is not None:
            config["max_tokens"] = properties.max_tokens
        if properties.reasoning_model is not None:
            config["reasoning"] = properties.reasoning_model

    if supports:
        if supports.images is not None:
            config["supports_images"] = supports.images
        if supports.files is not None:
            config["supports_files"] = supports.files
        if supports.videos is not None:
            config["supports_videos"] = supports.videos
        if supports.batch is not None:
            config["supports_batch"] = supports.batch
        if supports.temperature is not None:
            config["supports_temperature"] = supports.temperature
        if supports.tools is not None:
            config["supports_tools"] = supports.tools
    else:
        raise Exception(f"{registry_config.label} has no supports")

    # load provider config with correct type
    if provider_properties:
        ModelClass: type[LLM] = get_provider_registry()[registry_config.provider_name]
        if hasattr(ModelClass, "provider_config"):
            ProviderConfigClass: type[ProviderConfig] = type(ModelClass.provider_config)  # type: ignore
            provider_config: ProviderConfig = ProviderConfigClass.model_validate(
                provider_properties.model_dump(
                    exclude_unset=True, exclude_defaults=True
                )
            )
            config["provider_config"] = provider_config

    # load defaults
    config.update(defaults.model_dump(exclude_unset=True))

    loaded_config = LLMConfig(**config)

    # override only with explicitly set fields from override_config
    if override_config:
        loaded_config = loaded_config.model_copy(
            update=override_config.model_dump(exclude_unset=True)
        )
        # copy provider config with correct type
        if override_config.provider_config:
            loaded_config.provider_config = override_config.provider_config

    return loaded_config


def _get_model_from_registry(
    registry_config: ModelConfig, override_config: LLMConfig | None
) -> LLM:
    """
    Utility to return a model class from a registry entry.
    """
    model_config: LLMConfig = create_config(registry_config, override_config)
    model_config.registry_key = registry_config.full_key

    provider_name: str = registry_config.provider_name
    provider_endpoint: str = registry_config.provider_endpoint
    ModelClass: type[LLM] = get_provider_registry()[provider_name]

    return ModelClass(
        model_name=provider_endpoint,
        provider=registry_config.provider_name,
        config=model_config,
    )


def get_registry_config(model_str: str) -> ModelConfig | None:
    config = get_model_registry().get(model_str, None)
    return config


def get_registry_model(model_str: str, override_config: LLMConfig | None = None) -> LLM:
    """Get a model including default config"""
    registry_config = get_registry_config(model_str)
    if not registry_config:
        raise Exception(f"Model {model_str} not found in registry")

    return _get_model_from_registry(registry_config, override_config)


def get_raw_model(model_str: str, config: LLMConfig | None = None) -> LLM:
    """Get a model exluding default config"""
    provider, model_name = model_str.split("/", 1)
    ModelClass = get_provider_registry()[provider]
    return ModelClass(model_name=model_name, provider=provider, config=config)


@cache
def get_model_cost(model_str: str) -> CostProperties | None:
    model_config = get_model_registry().get(model_str)
    if not model_config:
        raise Exception(f"Model {model_str} not found in registry")
    return model_config.costs_per_million_token


class TokenDict(TypedDict, total=False):
    """Token counts for cost calculation."""

    in_tokens: int
    out_tokens: int
    reasoning_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None


async def recompute_cost(
    model_str: str,
    tokens: TokenDict,
) -> QueryResultCost:
    """
    Recompute the cost for a model based on token information.

    Uses the model provider's existing _calculate_cost method to ensure
    provider-specific cost calculations are applied.

    Args:
        model_str: The model identifier (e.g., "openai/gpt-4o")
        tokens: Dictionary containing token counts with keys:
            - in_tokens (required): Number of input tokens
            - out_tokens (required): Number of output tokens
            - reasoning_tokens (optional): Number of reasoning tokens
            - cache_read_tokens (optional): Number of cache read tokens
            - cache_write_tokens (optional): Number of cache write tokens

    Returns:
        QueryResultCost with computed costs

    Raises:
        ValueError: If required token parameters are missing
        Exception: If model not found in registry or costs not configured
    """
    if "in_tokens" not in tokens:
        raise ValueError("Token dict must contain 'in_tokens'")
    if "out_tokens" not in tokens:
        raise ValueError("Token dict must contain 'out_tokens'")

    model = get_registry_model(model_str)

    metadata = QueryResultMetadata(
        in_tokens=tokens["in_tokens"],
        out_tokens=tokens["out_tokens"],
        reasoning_tokens=tokens.get("reasoning_tokens"),
        cache_read_tokens=tokens.get("cache_read_tokens"),
        cache_write_tokens=tokens.get("cache_write_tokens"),
    )

    cost = await model._calculate_cost(metadata)  # type: ignore[arg-type]
    if cost is None:
        raise Exception(f"No cost information available for model {model_str}")

    return cost


@cache
def get_provider_names() -> list[str]:
    """Return all provider names in the registry"""
    return sorted([provider_name for provider_name in get_provider_registry().keys()])


@cache
def get_model_names() -> list[str]:
    """Return all model names in the registry"""
    return sorted([model_name for model_name in get_model_registry().keys()])


@cache
def get_model_names_by_provider(provider_name: str) -> list[str]:
    """Return all models in the registry from a provider"""
    return [
        model.full_key
        for model in get_model_registry().values()
        if model.provider_name.lower() == provider_name.lower()
    ]


@cache
def _get_tiktoken_encoder():
    """Get cached tiktoken encoder for consistent tokenization."""
    return tiktoken.encoding_for_model("gpt-4o")


def auto_trim_document(
    model_name: str,
    document: str,
) -> str:
    """
    Automatically trim document to fit within model's context window,
    leaving a buffer for instructions and output.

    Args:
        model_name: The name of the model in the registry
        document: The document text to trim

    Returns:
        Trimmed document, or original document if trimming isn't needed
    """

    max_tokens = get_max_document_tokens(model_name)

    encoding = _get_tiktoken_encoder()
    tokens = encoding.encode(document)

    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        document = encoding.decode(tokens)

    return document


def get_max_document_tokens(model_name: str, output_buffer: int = 10000) -> int:
    """
    Get the maximum document tokens for a model by looking up its context window
    from the registry and subtracting a configurable buffer for instructions and output.

    Args:
        model_name: The name of the model in the registry
        output_buffer: Number of tokens to reserve for output (default 10000)

    Returns:
        Maximum tokens to use for documents
    """
    # Import here to avoid circular imports
    from model_library.utils import get_context_window_for_model

    context_window = get_context_window_for_model(model_name)
    return context_window - output_buffer
