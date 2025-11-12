from model_library.settings import ModelLibrarySettings
from model_library.base import LLM, LLMConfig

model_library_settings: ModelLibrarySettings = ModelLibrarySettings()


def model(model_str: str, override_config: LLMConfig | None = None) -> LLM:
    from model_library.registry_utils import get_registry_model

    return get_registry_model(model_str, override_config)


def raw_model(model_str: str, override_config: LLMConfig | None = None) -> LLM:
    from model_library.registry_utils import get_raw_model

    return get_raw_model(model_str, config=override_config)


__all__ = [
    "model_library_settings",
    "model",
    "raw_model",
]
