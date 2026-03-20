import json
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
    @override
    def _get_default_api_key(self) -> str:
        return json.dumps(
            {
                "AZURE_API_KEY": model_library_settings.AZURE_API_KEY,
                "AZURE_ENDPOINT": model_library_settings.AZURE_ENDPOINT,
                "AZURE_API_VERSION": model_library_settings.get(
                    "AZURE_API_VERSION", "2025-04-01-preview"
                ),
            }
        )

    @override
    def get_client(self, api_key: str | None = None) -> AsyncAzureOpenAI:
        if not self.has_client():
            assert api_key
            creds = json.loads(api_key)
            client = AsyncAzureOpenAI(
                api_key=creds["AZURE_API_KEY"],
                azure_endpoint=creds["AZURE_ENDPOINT"],
                api_version=creds["AZURE_API_VERSION"],
                http_client=default_httpx_client(),
                max_retries=3,
            )
            self.assign_client(client)
        return super(OpenAIModel, self).get_client(api_key)

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
