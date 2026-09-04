"""Model metadata route registration."""

from typing import Any

from fastapi import FastAPI

from model_library.register_models import (
    ModelConfig,
    get_deprecated_model_registry,
    get_model_registry,
    visible_registry_keys,
)
from model_library.registry_utils import get_model_names, get_registry_config


def register_model_routes(app: FastAPI) -> None:
    @app.get("/registry")
    async def registry_snapshot(
        include_deprecated: bool = False,
        include_alt_keys: bool = True,
        include_excluded_fields: bool = False,
    ):
        registry = get_model_registry()
        if include_deprecated:
            # Active entries win over retired ones sharing a key.
            registry = {**get_deprecated_model_registry(), **registry}
        visible_keys = visible_registry_keys(registry, include_alt_keys)
        exclude = None
        if not include_excluded_fields:
            exclude = {
                "supports": {"transcription"},
                "country": True,
                "rate_limit": True,
            }

        def serialize_config(config: ModelConfig) -> dict[str, Any]:
            payload = config.model_dump(mode="json", exclude=exclude)
            if include_excluded_fields:
                if config.rate_limit is None:
                    payload.pop("rate_limit")
                else:
                    payload["rate_limit"] = config.rate_limit.model_dump(
                        mode="json", exclude_unset=True
                    )
            return payload

        return {
            "models": {
                key: serialize_config(config)
                for key, config in registry.items()
                if key in visible_keys
            }
        }

    @app.get("/models")
    async def list_models():
        models: list[dict[str, Any]] = []
        for name in get_model_names():
            config = get_registry_config(name)
            if config is None:
                continue
            models.append(
                {
                    "id": name,
                    "supports_tools": config.supports.tools or False,
                    "supports_images": config.supports.images or False,
                    "supports_files": config.supports.files or False,
                    "supports_audio": config.supports.audio or False,
                    "supports_transcription": config.supports.transcription or False,
                    "supports_structured_output": config.supports.output_schema
                    or False,
                }
            )
        return models
