import io
from typing import Any, Literal, Sequence, cast

from together import AsyncTogether
from together.types.chat_completions import (
    ChatCompletionMessage,
    ChatCompletionResponse,
)
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    LLM,
    FileInput,
    FileWithBase64,
    FileWithId,
    FileWithUrl,
    InputItem,
    LLMConfig,
    QueryResult,
    QueryResultCost,
    QueryResultMetadata,
    TextInput,
    ToolDefinition,
)
from model_library.exceptions import (
    BadInputError,
    MaxOutputTokensExceededError,
    ModelNoOutputError,
)
from model_library.file_utils import trim_images
from model_library.model_utils import get_reasoning_in_tag
from model_library.providers.openai import OpenAIModel
from model_library.utils import create_openai_client_with_defaults


class TogetherModel(LLM):
    _client: AsyncTogether | None = None

    @override
    def get_client(self) -> AsyncTogether:
        if not TogetherModel._client:
            TogetherModel._client = AsyncTogether(
                api_key=model_library_settings.TOGETHER_API_KEY,
            )
        return TogetherModel._client

    def __init__(
        self,
        model_name: str,
        provider: Literal["together"] = "together",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # https://docs.together.ai/docs/openai-api-compatibility
        self.delegate: OpenAIModel | None = (
            None
            if self.native
            else OpenAIModel(
                model_name=model_name,
                provider=provider,
                config=config,
                custom_client=create_openai_client_with_defaults(
                    api_key=model_library_settings.TOGETHER_API_KEY,
                    base_url="https://api.together.xyz/v1",
                ),
                use_completions=False,
            )
        )

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
    ) -> list[dict[str, Any] | Any]:
        new_input: list[dict[str, Any] | Any] = []
        content_user: list[dict[str, Any]] = []

        def flush_content_user():
            nonlocal content_user

            if content_user:
                new_input.append({"role": "user", "content": content_user})
                content_user = []

        for item in input:
            match item:
                case TextInput():
                    content_user.append({"type": "text", "text": item.text})
                case FileWithBase64() | FileWithUrl() | FileWithId():
                    match item.type:
                        case "image":
                            content_user.append(await self.parse_image(item))
                        case "file":
                            content_user.append(await self.parse_file(item))
                case ChatCompletionMessage():
                    flush_content_user()
                    new_input.append(item)
                case _:
                    raise BadInputError("Unsupported input type")

        flush_content_user()

        return new_input

    @override
    async def parse_image(
        self,
        image: FileInput,
    ) -> dict[str, Any]:
        match image:
            case FileWithBase64():
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image.mime};base64,{image.base64}"
                    },
                }
            case _:
                # docs show that we can pass in s3 location somehow
                raise BadInputError("Unsupported image type")

    @override
    async def parse_file(
        self,
        file: FileInput,
    ) -> Any:
        raise NotImplementedError()

    @override
    async def parse_tools(
        self,
        tools: list[ToolDefinition],
    ) -> Any:
        raise NotImplementedError()

    @override
    async def upload_file(
        self,
        name: str,
        mime: str,
        bytes: io.BytesIO,
        type: Literal["image", "file"] = "file",
    ) -> FileWithId:
        raise NotImplementedError()

    @override
    async def _query_impl(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        **kwargs: object,
    ) -> QueryResult:
        if self.delegate:
            return await self.delegate_query(input, tools=tools, **kwargs)

        # llama supports max 5 images
        if "lama-4" in self.model_name:
            input = trim_images(input, max_images=5)

        messages: list[dict[str, Any]] = []

        if "nemotron-super" in self.model_name:
            # move system prompt to prompt
            if "system_prompt" in kwargs:
                first_text_item = next(
                    (item for item in input if isinstance(item, TextInput)), None
                )
                if not first_text_item:
                    raise Exception(
                        "Given system prompt for nemotron-super model, but no text input found"
                    )
                system_prompt = kwargs.pop("system_prompt")
                first_text_item.text = f"SYSTEM PROMPT: {system_prompt}\nUSER PROMPT: {first_text_item.text}"

            # set system prompt to detailed thinking
            mode = "on" if self.reasoning else "off"
            kwargs["system_prompt"] = f"detailed thinking {mode}"
            messages.append(
                {
                    "role": "system",
                    "content": f"detailed thinking {mode}",
                }
            )

        if "system_prompt" in kwargs:
            messages.append({"role": "system", "content": kwargs.pop("system_prompt")})

        messages.extend(await self.parse_input(input))

        body: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "model": self.model_name,
            "messages": messages,
        }

        if self.supports_temperature:
            if self.temperature is not None:
                body["temperature"] = self.temperature
            if self.top_p is not None:
                body["top_p"] = self.top_p

        body.update(kwargs)

        response = await self.get_client().chat.completions.create(**body, stream=False)  # pyright: ignore[reportAny]

        response = cast(ChatCompletionResponse, response)

        if not response or not response.choices or not response.choices[0].message:
            raise ModelNoOutputError("Model returned no completions")

        text = str(response.choices[0].message.content)
        reasoning = None

        if response.choices[0].finish_reason == "length" and not text:
            raise MaxOutputTokensExceededError()

        if self.reasoning:
            text, reasoning = get_reasoning_in_tag(text)

        output = QueryResult(
            output_text=text,
            reasoning=reasoning,
            history=[*input, response.choices[0].message],
        )

        if response.usage:
            output.metadata.in_tokens = response.usage.prompt_tokens
            output.metadata.out_tokens = response.usage.completion_tokens
            # no cache tokens it seems
        return output

    @override
    async def _calculate_cost(
        self,
        metadata: QueryResultMetadata,
        batch: bool = False,
        bill_reasoning: bool = True,
    ) -> QueryResultCost | None:
        # https://docs.together.ai/docs/dedicated-inference#prompt-caching
        # By default, caching is not enabled. To turn on prompt caching, remove --no-prompt-cache from the create command

        # https://docs.together.ai/docs/inference-faqs#can-i-cache-prompts-or-use-speculative-decoding%3F
        # TODO: Together supports optimizations like prompt caching and speculative decoding for models that allow it, reducing latency and improving throughput.
        return await super()._calculate_cost(metadata, batch, bill_reasoning=True)
