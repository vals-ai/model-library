import io
from typing import Literal

import httpx
from pydantic import SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    FileWithId,
    LLMConfig,
    ProviderConfig,
)
from model_library.base.query_ids import PromptCacheKeyMode
from model_library.providers.openai import OpenAIConfig
from model_library.register_models import register_provider
from model_library.utils import default_httpx_client


class MetaConfig(ProviderConfig):
    use_responses: bool = False
    prompt_cache_key: PromptCacheKeyMode | None = None


@register_provider("meta")
class MetaModel(DelegateOnly):
    provider_config = MetaConfig()

    def __init__(
        self,
        model_name: str,
        provider: Literal["meta"] = "meta",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        base_url = "https://api.llama.com/v1"
        # https://docs.llama.com
        config = config or LLMConfig()
        delegate_config = config.model_copy(
            update={
                "custom_endpoint": config.custom_endpoint or base_url,
                "custom_api_key": config.custom_api_key
                or SecretStr(model_library_settings.META_API_KEY),
                "provider_config": OpenAIConfig(
                    prompt_cache_key=self.provider_config.prompt_cache_key,
                ),
            }
        )

        self.init_delegate(
            config=delegate_config,
            delegate_provider="openai",
            use_completions=not self.provider_config.use_responses,
        )

    @override
    async def upload_file(
        self,
        name: str,
        mime: str,
        bytes: io.BytesIO,
        type: Literal["image", "file"] = "file",
    ) -> FileWithId:
        client = default_httpx_client()
        client.base_url = httpx.URL("https://api.llama.com/v1")
        client.headers["Authorization"] = (
            f"Bearer {model_library_settings.META_API_KEY}"
        )

        file_bytes = bytes.getvalue()

        # step 1: create upload
        upload_resp = await client.post(
            "/uploads",
            json={
                "bytes": len(file_bytes),
                "filename": name,
                "mime_type": mime,
                "purpose": "ephemeral_attachment",
            },
        )
        upload_resp.raise_for_status()
        upload_id: str = upload_resp.json()["id"]

        # step 2: upload file data
        file_resp = await client.post(
            f"/uploads/{upload_id}",
            files={"data": (name, file_bytes, mime)},
        )
        file_resp.raise_for_status()
        file_id: str = file_resp.json()["file_id"]

        return FileWithId(
            type=type,
            name=name,
            mime=mime,
            file_id=file_id,
        )
