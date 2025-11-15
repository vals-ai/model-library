from typing import Literal

from openai.lib.azure import AsyncAzureOpenAI
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    LLMConfig,
)
from model_library.providers.openai import OpenAIModel
from model_library.register_models import register_provider
from model_library.utils import default_httpx_client


@register_provider("azure")
class AzureOpenAIModel(OpenAIModel):
    _azure_client: AsyncAzureOpenAI | None = None

    @override
    def get_client(self) -> AsyncAzureOpenAI:
        if not AzureOpenAIModel._azure_client:
            AzureOpenAIModel._azure_client = AsyncAzureOpenAI(
                api_key=model_library_settings.AZURE_API_KEY,
                azure_endpoint=model_library_settings.AZURE_ENDPOINT,
                api_version=model_library_settings.get(
                    "AZURE_API_VERSION", "2025-04-01-preview"
                ),
                http_client=default_httpx_client(),
                max_retries=1,
            )
        return AzureOpenAIModel._azure_client

    def __init__(
        self,
        model_name: str,
        provider: Literal["azure"] = "azure",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(
            model_name=model_name,
            provider=provider,
            config=config,
            use_completions=False,
        )
