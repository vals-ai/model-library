from typing import Literal

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    LLMConfig,
)
from model_library.register_models import register_provider


@register_provider("zai")
class ZAIModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["zai"] = "zai",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://docs.z.ai/guides/develop/openai/python
        self.init_delegate(
            config=config,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            api_key=model_library_settings.ZAI_API_KEY,
            use_completions=True,
            delegate_provider="openai",
        )
