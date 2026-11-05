import json
import re
from typing import Literal

from openai.lib.azure import AsyncAzureOpenAI
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    LLMConfig,
    QueryResultMetadata,
)
from model_library.base.router import RouterLLM
from model_library.providers.openai import OpenAIModel, completions
from model_library.register_models import register_provider
from model_library.registry_utils import get_registry_config
from model_library.utils import default_httpx2_client

ROUTING_FEE_PER_M_INPUT_TOKENS = 0.14
SERVED_MODEL_PROVIDERS = ("azure", "openai", "anthropic", "grok")
DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def _served_model_key(served_model: str) -> str | None:
    for name in (served_model, DATE_SUFFIX.sub("", served_model)):
        for provider in SERVED_MODEL_PROVIDERS:
            if get_registry_config(key := f"{provider}/{name}"):
                return key
    return None


@register_provider("azure")
class AzureOpenAIModel(OpenAIModel, RouterLLM):
    # The key travels with an endpoint and API version, not as one setting.
    api_key_setting = None

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
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> AsyncAzureOpenAI:
        if not self.has_client():
            assert api_key
            creds = json.loads(api_key)
            client = AsyncAzureOpenAI(
                api_key=creds["AZURE_API_KEY"],
                azure_endpoint=base_url or creds["AZURE_ENDPOINT"],
                api_version=creds["AZURE_API_VERSION"],
                http_client=default_httpx2_client(),
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
        registry_config = get_registry_config(self._registry_key or "")
        if registry_config and registry_config.metadata.router:
            self._api = completions
            self.router_mode = True

    @override
    def _served_model_cost_key(self, served_model: str) -> str | None:
        return _served_model_key(served_model)

    @override
    def _router_fee(self, metadata: QueryResultMetadata) -> float:
        return ROUTING_FEE_PER_M_INPUT_TOKENS * metadata.total_input_tokens / 1_000_000
