from model_library.base import LLM, LLMConfig
from model_library.logging import set_logging
from model_library.settings import ModelLibrarySettings

model_library_settings: ModelLibrarySettings = ModelLibrarySettings()

set_logging()


def model(model_str: str, override_config: LLMConfig | None = None) -> LLM:
    from model_library.registry_utils import get_registry_model

    return get_registry_model(model_str, override_config)


def raw_model(model_str: str, config: LLMConfig | None = None) -> LLM:
    from model_library.registry_utils import get_raw_model

    return get_raw_model(model_str, config=config)


__all__ = [
    "model_library_settings",
    "model",
    "raw_model",
    "set_logging",
]
