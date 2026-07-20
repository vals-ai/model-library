"""Model metadata route registration."""

from typing import Any

from fastapi import FastAPI

from model_library.registry_utils import get_model_names, get_registry_config
from model_library.register_models import get_model_registry


def register_model_routes(app: FastAPI) -> None:
    @app.get("/registry")
    async def registry_snapshot():
        registry = get_model_registry()
        return {
            "models": {
                key: config.model_dump(mode="json") for key, config in registry.items()
            }
        }

    @app.get("/models")
    async def list_models():
        models: list[dict[str, Any]] = []
        for name in get_model_names():
            config = get_registry_config(name)
            if not config or not config.supports:
                continue
            models.append(
                {
                    "id": name,
                    "supports_tools": config.supports.tools or False,
                    "supports_images": config.supports.images or False,
                    "supports_files": config.supports.files or False,
                    "supports_audio": config.supports.audio or False,
                    "supports_structured_output": config.supports.output_schema
                    or False,
                }
            )
        return models
