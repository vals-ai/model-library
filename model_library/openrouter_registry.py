from typing import Any

import httpx

from model_library.register_models import (
    CacheCost,
    CostProperties,
    DefaultParameters,
    ModelConfig,
    Properties,
    Supports,
)
from model_library.utils import get_logger

logger = get_logger("openrouter_registry")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"

_openrouter_model_configs: dict[str, ModelConfig] = {}
_fetched = False

MILLION = 1_000_000


def _fetch_openrouter_models() -> dict[str, ModelConfig]:
    global _fetched
    if not _fetched:
        try:
            response = httpx.get(OPENROUTER_API_URL, timeout=30)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            for m in data.get("data", []):
                openrouter_id: str = m["id"]
                try:
                    _openrouter_model_configs[openrouter_id] = _build_model_config(
                        openrouter_id, m
                    )
                except Exception:
                    logger.warning(
                        f"Failed to parse OpenRouter model {openrouter_id}",
                        exc_info=True,
                    )
            logger.debug(
                f"Fetched {len(_openrouter_model_configs)} models from OpenRouter"
            )
            _fetched = True
        except Exception:
            logger.warning("Failed to fetch OpenRouter models", exc_info=True)
    return _openrouter_model_configs


def _build_model_config(openrouter_id: str, api_data: dict[str, Any]) -> ModelConfig:
    architecture: dict[str, Any] = api_data.get("architecture", {})
    input_modalities: list[str] = architecture.get("input_modalities", ["text"])
    supported_params: list[str] = api_data.get("supported_parameters", [])
    pricing: dict[str, str] = api_data.get("pricing", {})
    api_defaults: dict[str, Any] = api_data.get("default_parameters") or {}

    # NOTE: all of these below config values are specific to the actual model provider that OpenRouter routes to.
    # unfortunately, we don't actually support routing to different providers, so we can only use
    # the values from the OpenRouter API which seems to default to a specific provider

    input_cost = float(pricing.get("prompt", "0")) * MILLION
    output_cost = float(pricing.get("completion", "0")) * MILLION

    cache: CacheCost | None = None
    cache_read_str = pricing.get("input_cache_read")
    cache_write_str = pricing.get("input_cache_write")
    if cache_read_str or cache_write_str:
        cache = CacheCost(
            read=float(cache_read_str) * MILLION if cache_read_str else None,
            write=float(cache_write_str) * MILLION if cache_write_str else None,
        )

    context_window: int = api_data["context_length"]

    full_key = f"openrouter/{openrouter_id}"

    return ModelConfig(
        company=openrouter_id.split("/")[0],
        label=api_data.get("name", openrouter_id),
        open_source=False,
        properties=Properties(
            context_window=context_window,
            max_tokens=0,
            training_cutoff=None,
            reasoning_model="reasoning" in supported_params,
        ),
        supports=Supports(
            images="image" in input_modalities,
            videos="video" in input_modalities,
            files="file" in input_modalities,
            batch=False,
            temperature="temperature" in supported_params,
            tools="tools" in supported_params,
            output_schema="response_format" in supported_params,
        ),
        costs_per_million_token=CostProperties(
            input=input_cost,
            output=output_cost,
            cache=cache,
        ),
        default_parameters=DefaultParameters(
            temperature=api_defaults.get("temperature"),
            top_p=api_defaults.get("top_p"),
            top_k=api_defaults.get("top_k"),
        ),
        provider_endpoint=openrouter_id,
        provider_name="openrouter",
        full_key=full_key,
        slug=full_key.replace("/", "_"),
    )


def resolve_openrouter_model(model_str: str) -> ModelConfig | None:
    openrouter_id = model_str[len("openrouter/") :]
    models = _fetch_openrouter_models()
    return models.get(openrouter_id)
