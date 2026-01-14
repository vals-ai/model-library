from typing import Any, Literal, Sequence

from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    InputItem,
    LLMConfig,
    ToolDefinition,
)
from model_library.providers.openai import OpenAIModel
from model_library.register_models import register_provider
from model_library.utils import default_httpx_client, create_openai_client_with_defaults


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
        self.delegate = OpenAIModel(
            model_name=self.model_name,
            provider=self.provider,
            config=config,
            custom_client=create_openai_client_with_defaults(
                api_key=model_library_settings.ZAI_API_KEY,
                base_url="https://open.bigmodel.cn/api/paas/v4/",
            ),
            use_completions=True,
        )

    # WARNING: tokenizer predictions are inexact!
    @override
    async def count_tokens(
        self,
        input: Sequence[InputItem],
        *,
        history: Sequence[InputItem] = [],
        tools: list[ToolDefinition] = [],
        **kwargs: object,
    ) -> int:
        """
        Count tokens using ZAI's tokenizer API endpoint.
        https://docs.z.ai/api-reference/tools/tokenizer
        """
        # Build the request body using delegate's build_body
        assert self.delegate
        combined_input = [*history, *input]

        # special case: API fails on no msgs
        if not combined_input:
            return 0

        body = await self.delegate.build_body(combined_input, tools=tools, **kwargs)

        # Extract only the fields needed for the tokenizer API
        tokenizer_body: dict[str, Any] = {
            "model": body["model"],
            "messages": body["messages"],
        }

        # Add tools if present
        if "tools" in body and body["tools"]:
            tokenizer_body["tools"] = body["tools"]

        # Call ZAI tokenizer API
        async with default_httpx_client() as client:
            response = await client.post(
                "https://api.z.ai/api/paas/v4/tokenizer",
                json=tokenizer_body,
                headers={
                    "Authorization": f"Bearer {model_library_settings.ZAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )

            try:
                # checks response status code
                response.raise_for_status()

                # extracts token count from response
                result = response.json()
                return result["usage"]["total_tokens"]
            except Exception as e:
                self.logger.error(
                    "Tokenizer API error: %s",
                    e,
                    "Falling back to delegate's count_tokens method",
                )
                return await super().count_tokens(
                    input, history=history, tools=tools, **kwargs
                )
