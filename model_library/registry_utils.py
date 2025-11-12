from functools import cache
from pathlib import Path

import tiktoken

from model_library.base import LLM, LLMConfig, ProviderConfig
from model_library.register_models import (
    MAPPING_PROVIDERS,
    CostProperties,
    ModelConfig,
    get_model_registry,
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
    class_properties = registry_config.class_properties
    provider_properties = registry_config.provider_properties
    defaults = registry_config.default_parameters

    if properties:
        if properties.max_token_output is not None:
            config["max_tokens"] = properties.max_token_output
        if properties.reasoning_model is not None:
            config["reasoning"] = properties.reasoning_model

    if class_properties:
        if class_properties.supports_images is not None:
            config["supports_images"] = class_properties.supports_images
        if class_properties.supports_files is not None:
            config["supports_files"] = class_properties.supports_files
        if class_properties.supports_videos is not None:
            config["supports_videos"] = class_properties.supports_videos
        if class_properties.supports_batch_requests is not None:
            config["supports_batch"] = class_properties.supports_batch_requests
        if class_properties.supports_temperature is not None:
            config["supports_temperature"] = class_properties.supports_temperature
        if class_properties.supports_tools is not None:
            config["supports_tools"] = class_properties.supports_tools

    # load provider config with correct type
    if provider_properties:
        ModelClass: type[LLM] = MAPPING_PROVIDERS[registry_config.provider_name]
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
    ModelClass: type[LLM] = MAPPING_PROVIDERS[provider_name]

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
    ModelClass = MAPPING_PROVIDERS[provider]
    return ModelClass(model_name=model_name, provider=provider, config=config)


@cache
def get_model_cost(model_str: str) -> CostProperties | None:
    model_config = get_model_registry().get(model_str)
    if not model_config:
        raise Exception(f"Model {model_str} not found in registry")
    return model_config.costs_per_million_token


@cache
def get_provider_names() -> list[str]:
    """Return all provider names in the registry"""
    return sorted([provider_name for provider_name in MAPPING_PROVIDERS.keys()])


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
