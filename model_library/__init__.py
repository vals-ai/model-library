from __future__ import annotations

from model_library.logging import set_logging
from model_library.settings import ModelLibrarySettings

model_library_settings: ModelLibrarySettings = ModelLibrarySettings()

from model_library.base import LLM, LLMConfig  # noqa: E402

from model_library.custom_register_models import (  # noqa: E402
    load_custom_model_configs,
    load_latest_vals_model_configs,
)

set_logging()


def model(
    model_str: str,
    override_config: LLMConfig | None = None,
) -> LLM:
    from model_library.registry_utils import get_registry_model

    return get_registry_model(model_str, override_config)


def raw_model(
    model_str: str,
    config: LLMConfig | None = None,
) -> LLM:
    from model_library.registry_utils import get_raw_model

    return get_raw_model(model_str, config=config)


__all__ = [
    "model_library_settings",
    "model",
    "raw_model",
    "set_logging",
    "load_custom_model_configs",
    "load_latest_vals_model_configs",
]
