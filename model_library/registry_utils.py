import logging
from functools import cache
from pathlib import Path
from typing import TypedDict

from model_library.base import (
    LLM,
    LLMConfig,
    ProviderConfig,
    QueryResultCost,
    QueryResultMetadata,
)
from model_library.register_models import (
    CostProperties,
    ModelConfig,
    get_model_registry,
    get_provider_registry,
)

logger = logging.getLogger("model_library")
ALL_MODELS_PATH = Path(__file__).parent / "config" / "all_models.json"


def _gateway_url() -> str | None:
    from model_library import model_library_settings

    return model_library_settings.get("MODEL_GATEWAY_URL", None)


def _raise_gateway_metadata_helper_error(helper_name: str) -> None:
    raise RuntimeError(
        f"{helper_name}() is local-registry only when MODEL_GATEWAY_URL is set. "
        "Use get_registry_model() and await model.ensure_metadata_loaded() for "
        "authoritative gateway metadata, or get_model_registry() only for "
        "explicit bulk discovery snapshots."
    )


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
        config["max_tokens"] = properties.max_tokens
        config["reasoning"] = properties.reasoning_model

    if supports:
        config["supports_images"] = supports.images
        config["supports_files"] = supports.files
        config["supports_videos"] = supports.videos
        config["supports_batch"] = supports.batch
        config["supports_temperature"] = supports.temperature
        config["supports_tools"] = supports.tools
        config["supports_output_schema"] = supports.output_schema
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
        # copy provider_config with correct type (model_dump flattens to dict)
        if override_config.provider_config:
            loaded_config.provider_config = override_config.provider_config

    return loaded_config


def _get_model_from_registry(
    registry_config: ModelConfig,
    override_config: LLMConfig | None,
) -> LLM:
    """
    Utility to return a model class from a registry entry.
    """
    model_config: LLMConfig = create_config(registry_config, override_config)
    model_config.registry_key = registry_config.full_key

    provider_name: str = registry_config.provider_name
    provider_endpoint: str = registry_config.provider_endpoint
    ModelClass: type[LLM] = get_provider_registry()[provider_name]

    llm = ModelClass(
        model_name=provider_endpoint,
        provider=registry_config.provider_name,
        config=model_config,
    )
    llm._metadata = registry_config.model_copy(deep=True)  # pyright: ignore[reportPrivateUsage]
    return llm


def get_registry_config(model_str: str) -> ModelConfig | None:
    if _gateway_url():
        _raise_gateway_metadata_helper_error("get_registry_config")

    config = get_model_registry().get(model_str, None)
    if config is not None:
        return config

    if model_str.startswith("openrouter/"):
        from model_library.openrouter_registry import resolve_openrouter_model

        return resolve_openrouter_model(model_str)

    return None


def get_registry_model(
    model_str: str,
    override_config: LLMConfig | None = None,
) -> LLM:
    """Get a model including default config.

    If MODEL_GATEWAY_URL is set, returns an unsynced GatewayLLM that routes through the gateway server.
    Gateway mode does not require client-side registry knowledge. Call
    await model.ensure_metadata_loaded() before reading gateway metadata fields.
    """
    from model_library import model_library_settings

    gateway_url = model_library_settings.get("MODEL_GATEWAY_URL", None)
    if gateway_url:
        from model_library.base.gateway import GatewayLLM

        logger.info(
            "MODEL_GATEWAY_URL is set, routing through gateway: %s", gateway_url
        )
        provider, model_name = model_str.split("/", 1)
        return GatewayLLM(model_name, provider, config=override_config)

    registry_config = get_registry_config(model_str)
    if not registry_config:
        raise Exception(f"Model {model_str} not found in registry")

    if registry_config.provider_name in {"cursor", "devin"}:
        raise ValueError(
            f"Model {model_str} is only available through {registry_config.company} CLI"
        )

    return _get_model_from_registry(registry_config, override_config)


def get_raw_model(
    model_str: str,
    config: LLMConfig | None = None,
) -> LLM:
    """Get a model exluding default config"""
    provider, model_name = model_str.split("/", 1)
    ModelClass = get_provider_registry()[provider]
    return ModelClass(model_name=model_name, provider=provider, config=config)


def get_model_cost(model_str: str) -> CostProperties | None:
    if _gateway_url():
        _raise_gateway_metadata_helper_error("get_model_cost")

    return _get_model_cost_cached(model_str)


@cache
def _get_model_cost_cached(model_str: str) -> CostProperties | None:
    model_config = get_registry_config(model_str)
    if not model_config:
        raise Exception(f"Model {model_str} not found in registry")
    return model_config.costs_per_million_token


def get_model_input_context_window(model_name: str) -> int:
    """Return the input context window for the model"""
    if _gateway_url():
        _raise_gateway_metadata_helper_error("get_model_input_context_window")

    return _get_model_input_context_window_cached(model_name)


def get_input_context_window_from_config(model: ModelConfig) -> int:
    """Return usable input tokens from a registry config.

    OpenAI and Meta configs express a total context window that includes the
    output budget, so subtract max output tokens for prompt/input capacity.
    """
    context_window = model.properties.context_window
    if model.provider_name in {"openai", "meta"}:
        context_window -= model.properties.max_tokens
    return max(context_window, 0)


@cache
def _get_model_input_context_window_cached(model_name: str) -> int:
    model = get_registry_config(model_name)
    if not model:
        raise Exception(f"Model {model_name} not found in registry")

    return get_input_context_window_from_config(model)


class TokenDict(TypedDict, total=False):
    """Token counts for cost calculation."""

    in_tokens: int
    out_tokens: int
    reasoning_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None


def compute_model_cost(
    model_str: str,
    metadata: QueryResultMetadata,
    *,
    batch: bool = False,
    bill_reasoning: bool = True,
) -> QueryResultCost | None:
    costs = get_model_cost(model_str)
    if not costs:
        return None

    million = 1_000_000
    input_cost = costs.input
    output_cost = costs.output
    cache_read_cost, cache_write_cost = None, None

    if metadata.cache_read_tokens or metadata.cache_write_tokens:
        if not costs.cache:
            raise Exception("Cache costs not set")
        cache_read_cost, cache_write_cost = costs.cache.get_costs(input_cost)

    if costs.context and metadata.total_input_tokens > costs.context.threshold:
        input_cost, output_cost = costs.context.get_costs(
            input_cost,
            output_cost,
            metadata.total_input_tokens,
        )
        if costs.context.cache:
            cache_read_cost, cache_write_cost = costs.context.cache.get_costs(
                input_cost
            )

    if batch:
        if not costs.batch:
            raise Exception("Batch costs not set")
        input_cost, output_cost = costs.batch.get_costs(input_cost, output_cost)

    return QueryResultCost(
        input=input_cost * metadata.in_tokens / million,
        output=output_cost * metadata.out_tokens / million,
        reasoning=output_cost * metadata.reasoning_tokens / million
        if metadata.reasoning_tokens is not None and bill_reasoning
        else None,
        cache_read=cache_read_cost * metadata.cache_read_tokens / million
        if metadata.cache_read_tokens is not None and cache_read_cost
        else None,
        cache_write=cache_write_cost * metadata.cache_write_tokens / million
        if metadata.cache_write_tokens is not None and cache_write_cost
        else None,
    )


async def recompute_cost(
    model_str: str,
    tokens: TokenDict,
) -> QueryResultCost:
    """
    Recompute the cost for a model based on token information.

    Uses registry pricing only. This also supports models that are exposed
    through external CLIs and intentionally cannot instantiate a provider.

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

    metadata = QueryResultMetadata(
        in_tokens=tokens["in_tokens"],
        out_tokens=tokens["out_tokens"],
        reasoning_tokens=tokens.get("reasoning_tokens"),
        cache_read_tokens=tokens.get("cache_read_tokens"),
        cache_write_tokens=tokens.get("cache_write_tokens"),
    )

    cost = compute_model_cost(model_str, metadata)
    if cost is None:
        raise Exception(f"No cost information available for model {model_str}")
    return cost


@cache
def get_provider_names() -> list[str]:
    """Return all provider names in the registry"""
    return sorted([provider_name for provider_name in get_provider_registry().keys()])


def get_model_names(
    provider: str | None = None,
    include_deprecated: bool = False,
    include_alt_keys: bool = True,
) -> list[str]:
    """
    Return model names in the local registry.
    - provider: Filter by provider name
    - include_deprecated: Include deprecated models
    - include_alt_keys: Include alternative keys from the same provider
    """
    if _gateway_url():
        _raise_gateway_metadata_helper_error("get_model_names")

    return _get_model_names_cached(provider, include_deprecated, include_alt_keys)


@cache
def _get_model_names_cached(
    provider: str | None = None,
    include_deprecated: bool = False,
    include_alt_keys: bool = True,
) -> list[str]:
    registry = get_model_registry()
    alternative_keys_set: set[str] = set()

    if not include_alt_keys:
        for model in registry.values():
            for alt_item in model.alternative_keys:
                alt_key = (
                    alt_item if isinstance(alt_item, str) else list(alt_item.keys())[0]
                )
                if alt_key.split("/")[0] == model.provider_name:
                    alternative_keys_set.add(alt_key)

    return sorted(
        [
            model.full_key
            for model in get_model_registry().values()
            if (not provider or model.provider_name.lower() == provider.lower())
            and (not model.metadata.deprecated or include_deprecated)
            and model.full_key not in alternative_keys_set
        ]
    )
